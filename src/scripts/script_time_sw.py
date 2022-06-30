import os
import cv2
import numpy as np
import math
import glob
import time
import pandas as pd
import multiprocessing

def mutual_info_sw(n_couples, Ref_uint8, Flt_uint8, dim):
        j_h=np.histogram2d(Ref_uint8.ravel(),Flt_uint8.ravel(),bins=[256,256])[0]
        j_h=j_h/(n_couples*dim*dim)
          
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


for seed in (1234, 0, 98562):
    for n_couples in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        hist_dim = 256
        dim = 512
        t=0
        #args = parser.parse_args()
        #accel_number=args.thread_number
        #overlay = "./design_1_wrapper.bit"
        #clock = 100
        #thread_number = 1
        #accel_number = thread_number
        #platform = "Zynq"
        #caching = False
        image_dimension = 512
        #res_path = "./"
        #config = "ok"

        #iron = Overlay(overlay)
        #num_threads = accel_number
        #if platform=='Zynq':
        #    from pynq.ps import Clocks;
            #print("Previous Frequency "+str(Clocks.fclk0_mhz))
        #    Clocks.fclk0_mhz = clock; 
            #print("New frequency "+str(Clocks.fclk0_mhz))
        np.random.seed(seed)
        ref = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
        if seed in (1234, 0, 98562):
            flt = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
        elif seed in (73541, 3478, 87632):
            flt = ref
        else:
            flt = np.zeros((n_couples*image_dimension, image_dimension))
        #accel_list=mi_accel_map(iron, platform, caching, num_threads, image_dimension, config, n_couples)

         #time test single MI
        iterations=10
        t_tot = 0
        times=[]
        time_sw = []
        time_hw = []
        dim=image_dimension
        diffs=[]
        start_tot = time.time()
        print('seed: %d, ncouples: %d' % (seed, n_couples))
        for i in range(iterations):
            ref = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
            if seed in (1234, 0, 98562):
                flt = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
            elif seed in (73541, 3478, 87632):
                flt = ref
            else:
                flt = np.zeros((n_couples*image_dimension, image_dimension))
            start_single_sw = time.time()
            sw_mi=mutual_info_sw(n_couples, ref, flt, dim)
            end_single_sw = time.time()
            time_sw.append(end_single_sw - start_single_sw)
            #accel_list[0].prepare_ref_buff(ref)
            #accel_list[0].prepare_flt_buff(flt)
            start_single = time.time()
            #out = accel_list[0].exec_and_wait()
            end_single = time.time()
            #print("Hw res: "+str(out[0]))
            #print("Sw res: "+str(sw_mi))
            t = end_single - start_single
            times.append(t)
            time_hw.append(t)
            #diff=sw_mi - out[0]
            #diffs.append(diff)
            t_tot = t_tot +  t
            #iron.free()

        end_tot = time.time()

        #accel_list[0].reset_cma_buff()
        #print("Mean value of hw vs sw difference" +str(np.mean(diffs)))
        #with open('time_software_comp.csv', 'a') as software_file:
        #    ratio = np.divide(time_sw, time_hw)
        #    software_file.write("%d, %d, %s, %s, %s, %s,\n" % (n_couples, seed, np.mean(time_sw), np.std(time_sw), np.mean(ratio), np.std(ratio)))

        with open('time_temp.csv', 'a') as file:
            #ratio = np.divide(time_sw, time_hw)
            file.write("%d, %d,  %s, %s,\n" % \
                 (n_couples, seed, np.mean(time_sw), np.std(time_sw)))
        print("seed: %d, n_couples: %d" % (seed, n_couples))
