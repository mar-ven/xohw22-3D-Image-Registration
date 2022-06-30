#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import cv2
import numpy as np
import math
import glob
import time
import pandas as pd
import multiprocessing
from pynq import Overlay
import pynq
from pynq import allocate
import struct
import statistics
import argparse


# In[13]:


import pynq
from pynq import allocate


# In[14]:


# function for specific multicore mapping on different platforms, memory banks and namings
def mi_accel_map(iron_pl, platform, caching, num_threads=1, i_ref_sz=512, config=None, n_couples = 1):
    mi_list = []
    if(caching):
        ref_size=i_ref_sz
        ref_dt="uint8"
        flt_size=1
        flt_dt=np.float32
        mi_size=1
        mi_dt="u4"
    else:
        ref_size=i_ref_sz
        ref_dt="uint8"
        flt_size=i_ref_sz
        flt_dt="uint8"
        mi_size=1
        mi_dt=np.float32

    if(num_threads>=1):
        if platform == 'Alveo':#pcie card based
            mi_acc_0=SingleAccelMI(iron_pl.mutual_information_master_1_1, platform, iron_pl.bank0,                caching, ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples)
        else: #ZYNQ based
            mi_acc_0=SingleAccelMI(iron_pl.mutual_information_m_0, platform, None,                caching, ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples, config)
        mi_list.append(mi_acc_0)
    if (num_threads >= 2):
        if platform == 'Alveo':#pcie card based
            mi_acc_1=SingleAccelMI(iron_pl.mutual_information_master_2_1,platform, iron_pl.bank1,                caching, ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples)
        else: #ZYNQ based
            mi_acc_1=SingleAccelMI(iron_pl.mutual_information_m_1,platform,None,                caching, ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples, config)
        mi_list.append(mi_acc_1)
    if(num_threads >= 3):
        if platform == 'Alveo':#pcie card based
            mi_acc_2=SingleAccelMI(iron_pl.mutual_information_master_3_1,platform, iron_pl.bank2,                caching, ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples)
        else: #ZYNQ based
            mi_acc_2=SingleAccelMI(iron_pl.mutual_information_m_2,platform,None,                caching, ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples, config)
        mi_list.append(mi_acc_2)
    if(num_threads >= 4):
        if platform == 'Alveo':#pcie card based
            mi_acc_3=SingleAccelMI(iron_pl.mutual_information_master_4_1,platform, iron_pl.bank3,                caching, ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples)
        else: #ZYNQ based
            mi_acc_3=SingleAccelMI(iron_pl.mutual_information_m_3,platform,None,                caching, ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples, config)
        mi_list.append(mi_acc_3)
    return mi_list



# In[15]:


class SingleAccelMI :
    
    
###########################################################
# DEFAULTS of the INIT
###########################################################
#
# platform='Alveo'
#caching=False
#ref_size=512
# ref_dt="uint8"
# flt_size=512, then to the power of 2
#flt_dt="uint8"
# mi_size=1 then to the power of 2
# n_couples = 1
# mi_dt=np.float32
#
###########################################################

    def __init__(self, accel_id,  platform='Alveo', mem_bank=None, caching=False, ref_size=512, ref_dt="uint8", flt_size=512, flt_dt="uint8", mi_size=1, mi_dt=np.float32, n_couples = 1, config=None):
            self.AP_CTRL = 0x00
            self.done_rdy = 0x6
            self.ap_start = 0x1
            self.REF_ADDR = 0x18
            self.FLT_ADDR_OR_MI = 0x10
            self.MI_ADDR_OR_FUNCT = 0x20
            self.N_COUPLES_ADDR =0x28
            
            self.LOAD_IMG = 0
            self.COMPUTE = 1
            self.n_couples = n_couples
            self.buff1_img = allocate(n_couples*ref_size*ref_size, ref_dt, target=mem_bank)
            self.buff2_img_mi = allocate(n_couples*flt_size*flt_size, flt_dt, target=mem_bank)
            self.buff3_mi_status = allocate(mi_size, mi_dt, target=mem_bank)

            self.buff1_img_addr = self.buff1_img.device_address
            self.buff2_img_mi_addr = self.buff2_img_mi.device_address
            self.buff3_mi_status_addr = self.buff3_mi_status.device_address
            
            self.accel = accel_id
            
            self.platform = platform
            self.caching = caching
            self.config = config
            # print(self.accel)
            # print(self.platform)
            # print(self.caching)

    def get_config(self):
        return self.config

    def init_accel(self, Ref_uint8, Flt_uint8):
        self.prepare_ref_buff(Ref_uint8)
        if not self.caching:
            self.prepare_flt_buff(Flt_uint8)
    
    def load_caching(self):
        if self.platform == 'Alveo':
            self.accel.call(self.buff1_img, self.buff2_img_mi, self.LOAD_IMG, self.buff3_mi_status) 
        else: #ZYNQ-based
            self.execute_zynq(self.LOAD_IMG)
    
    def read_status(self):
        return self.accel.mmio.read(self.STATUS_ADDR)

    def prepare_ref_buff(self, Ref_uint8):
        self.buff1_img[:] = Ref_uint8.flatten()
        self.buff1_img.flush()#sync_to_device
        if not self.caching:
            return
        else:
            if self.platform != 'Alveo':
                self.accel.write(self.STATUS_ADDR, self.buff3_mi_status_addr)
            self.load_caching()
            self.buff2_img_mi.invalidate()#sync_from_device
            self.buff3_mi_status.invalidate()#sync_from_device

    
    def prepare_flt_buff(self, Flt_uint8):
        if not self.caching:
            self.buff2_img_mi[:] = Flt_uint8.flatten()
            self.buff2_img_mi.flush() #sync_to_device
        else:
            self.buff1_img[:] = Flt_uint8.flatten()
            self.buff1_img.flush()#sync_to_device

    def execute_zynq(self, mi_addr_or_funct):
        self.accel.write(self.REF_ADDR, self.buff1_img.device_address)
        self.accel.write(self.FLT_ADDR_OR_MI, self.buff2_img_mi.device_address)
        self.accel.write(self.MI_ADDR_OR_FUNCT, mi_addr_or_funct)
        self.accel.write(self.N_COUPLES_ADDR, self.n_couples)
        self.accel.write(self.AP_CTRL, self.ap_start)
        while(self.accel.mmio.read(0) & 0x4 != 0x4):
            pass
    
    def exec_and_wait(self):
        result = []
        if not self.caching:
            if self.platform == 'Alveo':
                self.accel.call(self.buff1_img, self.buff2_img_mi, self.buff3_mi_status, self.n_couples)
            else:# ZYNQ based
                self.execute_zynq(self.buff3_mi_status.device_address)
            self.buff3_mi_status.invalidate()#sync_from_device
            result.append(self.buff3_mi_status)
        else:
            if self.platform == 'Alveo':
                self.accel.call(self.buff1_img, self.buff2_img_mi, self.COMPUTE, self.buff3_mi_status, self.n_couples)
            else:# ZYNQ based
                self.execute_zynq(self.COMPUTE)
            self.buff2_img_mi.invalidate()#sync_from_device
            result.append(self.buff2_img_mi)
            self.buff3_mi_status.invalidate()#sync_from_device
            result.append(self.buff3_mi_status)
            result.append(self.n_couples)
        
        return result

    
    def reset_cma_buff(self):
	
        self.buff1_img.freebuffer() 
        self.buff2_img_mi.freebuffer()
        self.buff3_mi_status.freebuffer()
        del self.buff1_img 
        del self.buff2_img_mi
        del self.buff3_mi_status
    
    def mutual_info_sw(self, Ref_uint8, Flt_uint8, dim):
        j_h=np.histogram2d(Ref_uint8.ravel(),Flt_uint8.ravel(),bins=[256,256])[0]
        j_h=j_h/(self.n_couples*dim*dim)
          
        j_h1=j_h[np.where(j_h>0.000000000000001)]
        entropy=(np.sum(j_h1*np.log2(j_h1)))*-1

        href=np.sum(j_h,axis=0)
        hflt=np.sum(j_h,axis=1)     

        href=href[np.where(href>0.000000000000001)]
        eref=(np.sum(href*(np.log2(href))))*-1

        hflt=hflt[np.where(hflt>0.000000000000001)]
        eflt=(sum(hflt*(np.log2(hflt))))*-1

        mutualinfo=eref+eflt-entropy

        return(mutualinfo)


# In[23]:


def main():
    filename = "output.csv"
    with open(filename, 'w') as file:
        file.write("n_couples, seed, output_mi, mean_time_hw, std_time_hw, mean_error, std_error, mean_time_sw, std_time_sw,\n")

    for seed in (1234, 0, 98562, 73541, 3478, 87632, 45638, 2134, 77899):
        for n_couples in range(1, 512):
            hist_dim = 256
            dim = 512
            t=0
            #args = parser.parse_args()
            #accel_number=args.thread_number
            overlay = "./design_1_wrapper.bit"
            clock = 100
            thread_number = 1
            accel_number = thread_number
            platform = "Zynq"
            caching = False
            image_dimension = 512
            res_path = "./"
            config = "ok"


            iron = Overlay(overlay)
            num_threads = accel_number
            if platform=='Zynq':
                from pynq.ps import Clocks;
                #print("Previous Frequency "+str(Clocks.fclk0_mhz))
                Clocks.fclk0_mhz = clock; 
                #print("New frequency "+str(Clocks.fclk0_mhz))
            np.random.seed(seed)
            ref = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
            if seed in (1234, 0, 98562):
                flt = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
            elif seed in (73541, 3478, 87632):
                flt = ref
            else:
                flt = np.zeros((n_couples*image_dimension, image_dimension))
            accel_list=mi_accel_map(iron, platform, caching, num_threads, image_dimension, config, n_couples)

             #time test single MI
            iterations=10
            t_tot = 0
            times=[]
            time_sw = []
            time_hw = []
            dim=image_dimension
            diffs=[]
            start_tot = time.time()
            for i in range(iterations):
                ref = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
                if seed in (1234, 0, 98562):
                    flt = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
                elif seed in (73541, 3478, 87632):
                    flt = ref
                else:
                    flt = np.zeros((n_couples*image_dimension, image_dimension))
                start_single_sw = time.time()
                sw_mi=accel_list[0].mutual_info_sw(ref, flt, dim)
                end_single_sw = time.time()
                time_sw.append(end_single_sw - start_single_sw)
                accel_list[0].prepare_ref_buff(ref)
                accel_list[0].prepare_flt_buff(flt)
                start_single = time.time()
                out = accel_list[0].exec_and_wait()
                end_single = time.time()
                #print("Hw res: "+str(out[0]))
                #print("Sw res: "+str(sw_mi))
                t = end_single - start_single
                times.append(t)
                time_hw.append(t)
                diff=sw_mi - out[0]
                diffs.append(diff)
                t_tot = t_tot +  t
                #iron.free()
            
            print('seed: %d, ncouples: %d, output: %f' % (seed, n_couples, out[0]))

            end_tot = time.time()
            accel_list[0].reset_cma_buff()

            with open(filename, 'a') as file:
                file.write("%d, %d, %f, %s, %s, %s, %s, %s, %s,\n" % (n_couples, seed, out[0], np.mean(times), np.std(times), np.mean(diffs), np.std(diffs), np.mean(time_sw), np.std(time_sw)))

            iron.free()


            del iron
if __name__== "__main__":
    main()