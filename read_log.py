from collections import namedtuple
from pprint import pprint

# "jpeg_psnr,jpeg_ssim,our_ssim,our_q,jpeg_psnrhvs,png_size,model_number,our_size,filename,jpeg_vifp,jpeg_q,jpeg_msssim,our_psnrhvsm,jpeg_psnrhvsm,our_vifp,our_psnr,our_msssim,our_psnrhvs,jpeg_size"


def process_one(eg):
    value_map = {}
    eg_s = eg.split('|')

    metrics = "PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM".lower().split(' ')
    meta = "filename model_number png_size jpeg_size our_size jpeg_q our_q".split() 

    first_line = eg_s[0].strip().split()
    for index, m in enumerate(meta):
        value_map[m] = first_line[index]

    for typ, v in zip(['jpeg', 'our'], [eg_s[1], eg_s[2]]):
        for m, value in zip(metrics, v.strip().split()):
            value_map[typ + '_' + m] = float(value)
    return value_map


def process_log(filename):
    f = open(filename).read().splitlines()
    values = [process_one(l) for l in f]
    return values

def print_given_average_metrics(values, metric):
    total = len(values)
    avg = sum([float(v[metric]) for v in values])/total
    print metric, str(avg)

def print_all_average_metrics(values, silent=False):
    out = {}
    kk = values[0].keys()
    total = len(values)
    for k in kk:
        if k == 'filename': continue
        avg = sum([float(v[k]) for v in values])/total
        out[str(k)] = avg
        if not silent:
            print str(k), str(avg)
    return out

def pprint_metrics(avg_metrics):
    out = {}
    for k,v in avg_metrics.iteritems():
        model, metric = k.split('_')
        if metric not in out:
            out[metric] = {}
        out[metric][model] = v
    return out

def pprint_by_categories(values, metric=None, data_type='mit'):
    from itertools import product

    if data_type == 'mit':
        categories = json.load(open('./categories.json'))
    else:
        categories = ['01', '02', '03', '04', '05', '06', '07', '08', '09'] + map(str, range(10,25))

    
    out = []
    for cat in categories:
        filtered_values = filter(lambda x: cat in x['filename'], values) 
        avg_metrics = print_all_average_metrics(filtered_values, silent=True) 
        if metric is not None:
            res = pprint_metrics(avg_metrics)[metric]
        else:
            res =  pprint_metrics(avg_metrics)
        pprint ( res )
        out.append((cat,res))
    return out

def plot(values, metric_name):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys

    plt.style.use('ggplot')

    fig, ax = plt.subplots(1, 1, figsize=(25, 3))
    ax.margins(0)

    x = []
    y = []
    for index,v in enumerate( values ):
        # if not index: continue
        # plt.plot(x, new_recall, linewidth=2, label='Condensed Mem Network')
        x.append(index)
        y.append(v[1]['our']-v[1]['jpeg'])

    # plt.plot(x,y, 'o')
    # plt.semilogy(x,y)
    y_neg = [max(0,i) for i in y]
    y_pos = [min(0,i) for i in y]

    plt.bar(x,y_neg)
    plt.bar(x,y_pos, color='r')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    plt.title(metric_name.upper(), x=0.5, y=0.8, fontsize=14)
    plt.legend(loc='')
    ax.get_xaxis().set_visible(False)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    fig.tight_layout()
    # plt.savefig('plot_size_' + metric_name + '.png', bbox_inches='tight_layout', pad_inches=0)
    plt.savefig('plot_kodak_' + metric_name + '.png')


if __name__ == '__main__':
    import sys
    import json
    print_average    = True
    print_categories = False

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # filename = 'logs/log_small_ALL_model_69.log'
        filename = 'logs/log_small_100_model_6_79.log'
    values = process_log(filename)
    
    metrics = "PSNR SSIM MSSSIM VIFP PSNRHVS PSNRHVSM".lower().split(' ')
    
    if print_categories:
        for metric_name in metrics:
            out =  pprint_by_categories(values, metric_name, data_type='size')
            print metric_name, len(out)
            plot(out, metric_name)

    if print_average:
        avg_metrics = print_all_average_metrics(values, silent=True)
        avg_metrics = pprint_metrics(avg_metrics)
        pprint ( avg_metrics )
    print(filename, avg_metrics['ssim'])
    print_given_average_metrics(values, 'filename')
    print_given_average_metrics(values, 'our_q')
    print_given_average_metrics(values, 'jpeg_q')
    print_given_average_metrics(values, 'our_size')
    print_given_average_metrics(values, 'jpeg_size')
    print_given_average_metrics(values, 'jpeg_psnr')

    


    
    
