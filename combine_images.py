# from __future__ import division
import sys
import Image
import os
import numpy as np
# from array2pil import array2PIL
from util import load_image, array2PIL
import argparse
from scipy.stats import percentileofscore

parser = argparse.ArgumentParser()

parser.add_argument('-image'            , type=str , default= './images_2/ori_OutdoorNatural_073.png')
parser.add_argument('-map'              , type=str , default= './images_2/map_OutdoorNatural_073.png')
parser.add_argument('-output_directory' , type=str , default= 'temp')
parser.add_argument('-modifier'         , type=str , default= '')
parser.add_argument('-find_best'        , type=int , default=1)
parser.add_argument('-use_convert'      , type=int , default=1)
parser.add_argument('-jpeg_compression' , type=int , default=50)
parser.add_argument('-model'            , type=int , default=6)
parser.add_argument('-single'           , type=int , default=0)
parser.add_argument('-dataset'          , type=str , default='kodak')
args = parser.parse_args()


def make_quality_compression(ori,sal):
    print args.image,
    # if the size of the map is not the same original image, then blow it
    if ori.size != sal.size:
        sal = sal.resize(ori.size)

    sal_arr = np.asarray(sal)
    img_qualities = []
    quality_steps = [i*10 for i in xrange(1,11)]

    for q in quality_steps:
        name = 'temp/temp_' + str(q) + '.jpg'
        if args.use_convert:
            os.system('convert -colorspace sRGB -filter Lanczos -interlace Plane -type truecolor -quality ' + str(q) + ' ' + args.image + ' ' + name)
        else:
            ori.save(name, quality=q)
        img_qualities.append(np.asarray(Image.open(name)))
                   
    k = img_qualities[-1][:] # make sure it is a copy and not reference
    shape = k.shape 
    k.flags.writeable = True
    mx, mn = np.max(sal_arr), np.mean(sal_arr)
    sal_flatten = sal_arr.flatten()

    q_2,q_3,q_5,q_6,q_9 = map(lambda x: np.percentile(sal_arr, x), [20,30,50,60,90])

    q_a = [np.percentile(sal_arr, j) for j in quality_steps]
    low, med, high = 1, 5, 9

    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            for l in xrange(shape[2]):
                ss = sal_arr[i,j]

                if args.model == 1:
                    # model -1 
                    # hard-coded model
                    if ss > mn: qq = 9
                    else: qq = 6

                elif args.model == 2:
                    # model -2 
                    # linearly scaled technique
                    qq = (ss * 10 // mx) -1  + 3
                
                elif args.model == 3:
                    # model -3 
                    # percentile based technique
                    # qq = int(percentileofscore(sal_flatten, ss)/10)
                    for index, q_i in enumerate(q_a):
                        if ss < q_i: 
                            qq = index + 1
                            break

                elif args.model == 4:
                    # model -4 
                    # discrete percentile based technique
                    # if   ss < q_2: qq = 4 
                    if ss < q_2: qq = 4 
                    elif ss < q_6: qq = 6 
                    elif ss < q_9: qq = 8 
                    else: qq = 9

                elif args.model == 5:
                    # model -5
                    # two way percentile
                    if ss <  q_5: qq = 2
                    else: qq = 8

                elif args.model == 6:
                    # model -6
                    # two way percentile - higher coverage
                    if ss <  q_5: qq = 7
                    else: qq = 9
                    
                else:
                    raise Exception("unknown model number")

                if qq < low : qq = low
                if qq > high: qq = high 
                k[i,j,l] = img_qualities[qq][i,j,l]
                    
    # save the original file at the given quality level
    ori_compressed = args.output_directory + '/' + '_ORI_' + args.image.split('/')[-1] + '_' + str(args.jpeg_compression) + '.jpg'
    ori.save(ori_compressed, quality=args.jpeg_compression)
   
    
    ori_size = os.path.getsize(ori_compressed)
    os.system('convert ' + args.image + ' temp.png')
    uncompressed_size = os.path.getsize('temp.png')

    out_img = array2PIL(k)

    if args.find_best:
        out_name = args.output_directory + '/' + '_BEST_' + args.image.split('/')[-1] + '_' + '.jpg'
        for qual in xrange(90,20,-1):
            out_img.save(out_name, quality=qual)
            current_size = os.path.getsize(out_name)
            if current_size<= ori_size*1.02: 
                print args.model, uncompressed_size, ori_size, current_size, args.jpeg_compression, qual,' | ',
                break
        else:
            print args.model, uncompressed_size, ori_size, current_size, args.jpeg_compression, qual,' | ',

    else:
        final_quality = [100, 85, 65, 45]
        for fq in final_quality:
            out_name = args.output_directory + '/' + args.modifier + args.image.split('/')[-1] + '_' + str(fq) + '.jpg'
            out_img.save(out_name, quality=fq)
    return ori_compressed, out_name


# header
# filename, model_number, uncompressed_size, jpeg_size, current_size, jpeg_compression, current_compression,
# (JPEG) PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM
# (model) PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM

def get_metrics(ori, ori_compressed, out_name, size):
    
    metrics = "PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM".lower().split(' ')
    # TODO add -y (overwrite files) 
    # first convert all the three images to yuv format
    size_x = size[0] - size[0]%16 # this is to make sure we can get MS-SSIM 
    size_y = size[1] - size[1]%16 # metrics from VQMT, which requires divisible by 16

    # for x in [ori_compressed, out_name]:
    for x in [ori, ori_compressed, out_name]:
        yuv_convert_command = "ffmpeg -hide_banner -loglevel panic -y -i " + x +" -s " + str(size_x) + "x" + str(size_y) + " -pix_fmt yuv420p " + x +".yuv"
        os.system(yuv_convert_command)
        # print command
    for img_com in [ori_compressed, out_name]:
        command_metrics = "~/image_compression/vqmt " + \
                          ori+".yuv " + \
                          img_com+".yuv " + \
                          str(size_x) + " " + \
                          str(size_y) + " " + \
                          "1 1 out PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM"

        # print command_metrics
        os.system(command_metrics)
        for m in metrics:
            f = open('out_' + m + '.csv').read().splitlines()[1].split(',')[1]
            print f, 
        print ' | ',
    print ''

from glob import glob

if args.single:
    ori = Image.open(args.image)
    sal = Image.open(args.map)
    a,b = make_quality_compression(ori,sal)
    get_metrics(args.image,a,b, ori.size)

else:
    
    if args.dataset == 'kodak':
        image_path = '/home/ap/image_compression/kodak/*.png'
    elif args.dataset == 'large':
        image_path = 'images_directory/output_large/ori_*.png'
    else:
        assert Exception("Wrong dataset choosen")

    for image_file in glob(image_path):
        if args.dataset == 'large':
            map_file = 'images_directory/output_large/map' + image_file.split('/')[-1][3:-4] 
        elif args.dataset == 'kodak':
            map_file = 'images_directory/output_kodak/map_' + image_file.split('/')[-1] + '.jpg'
        args.image = image_file
        args.map   = map_file
        ori = Image.open(args.image)
        sal = Image.open(args.map)
        a,b = make_quality_compression(ori,sal)
        get_metrics(args.image,a,b, ori.size)
