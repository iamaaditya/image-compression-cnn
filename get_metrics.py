
# header
# filename, model_number, uncompressed_size, jpeg_size, current_size, jpeg_compression, current_compression,
# (JPEG) PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM
# (model) PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM

def get_metrics(original, compressed, out_name, size):
    
    metrics = "PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM".lower().split(' ')
    # TODO add -y (overwrite files) 
    # first convert all the three images to yuv format
    size_x = size[0] - size[0]%16 # this is to make sure we can get MS-SSIM 
    size_y = size[1] - size[1]%16 # metrics from VQMT, which requires divisible by 16

    for x in [original, compressed, out_name]:
        yuv_convert_command = "ffmpeg -hide_banner -loglevel panic -y -i " + x +" -s " + str(size_x) + "x" + str(size_y) + " -pix_fmt yuv420p " + x +".yuv"
        os.system(yuv_convert_command)
        # print command
    for img_com in [compressed, out_name]:
        command_metrics = "~/image_compression/vqmt " + \
                          original+".yuv " + \
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


