# Python code for question C103

import numpy as np
class Sorting(object):
    TEMP_FILE = 'temp_{0}.dat'
    
    def __init__(self,raw_file,memory_limit):
        self.memory_limit = memory_limit
        self.raw_file = raw_file
        self.id_list = []
        self.open_temp_files = []
        self.buffers = {}
        self.n_buffers = None
        self.buffers_pointer = None
        self.output_buffer = None
        self.buffer_size =  None
        self.number_count = {}
    
    def writeBlockFiles(self,data,temp_id):
        temp_file = self.TEMP_FILE.format(temp_id)
        f = open(temp_file,'wb')
        f.write(data)
        f.close()

    def splitRawFile(self):
        temp_id = 0
        raw_file = open(self.raw_file,'rb')
        while True:
            lines = raw_file.readlines(self.memory_limit)
            if lines == []:
                break
            self.bubbleSort(lines)
            self.writeBlockFile(lines,temp_id)
            self.id_list.append(temp_id)
            temp_id += 1    

    def openFiles(self):
        for i in range(len(self.id_list)):
            temp_file = self.TEMP_FILE.format(self.id_list[i])
            self.open_temp_files.append(open(temp_file,'rb',self.buffer_size))
    def closeFiles(self):
        for f in self.open_temp_files:
            f.close()
    
    def bubbleSort(self,seq):
        n = len(seq)
        for i in range(n-1):
            for j in range(n-1-i):
                if seq[j] > seq[j+1]:
                    seq[j],seq[j+1] = seq[j+1],seq[j]


    def initializeBuffers(self):
        _len = len(self.id_list)
        self.n_buffers = _len
        self.buffer_size = self.memory_limit / (_len*2)
        self.openFiles()
        for temp_id in self.id_list:
            self.buffers[temp_id] = self.open_temp_files[temp_id].readline()
        self.buffers_pointer = np.zeros(self.buffer_size)
        self.empty = set()

    def sort(self,output_filename):
        output_f = open(output_filename,'wb')
        self.initializeBuffers()
        compare_numbers = np.zeros(self.n_buffers)
        for temp_id in self.id_list:
            compare_numbers[temp_id] = self.buffers[temp_id][0]

        output_buffer = []
        while True():
            min_num = compare_numbers.min()
            n_buf = np.where(compare_numbers == min_num)[0][0]
            output_buffer.append(min_num)
            self.count(min_num)
            if len(output_buffer) >= self.buffer_size:
                output_f.wirte(''.join(n) for n in output_buffer)
                output_buffer = []
            _last_pointer = self.buffers_pointer[n_buf]
            if _last_pointer >= self.buffer_size -1:
                lines = self.open_temp_files[n_buf].readline()
                if lines == []:
                    self.empty.add(n_buf)
                    compare_numbers[n_buf] = 100000
                else:
                    self.buffers[n_buf] = lines
                    self.buffers_pointer[n_buf] = 0
                    compare_numbers[n_buf] = self.buffers[n_buf][0]
            else:
                self.buffers_pointer[n_buf] = _last_pointer + 1
                compare_numbers[n_buf] = self.buffers[n_buf][_last_pointer+1]
            if len(self.empty) == self.n_buffers:
                break
        if output_buffer !=  []:
            output_f.write(''.join(n) for n in output_buffer)
        self.closeFiles()
        output_f.close()

    def count(self,number):
        if number in self.number_count:
            self.number_count[number] += 1
        else:
            self.number_count[number] = 1



                

            


