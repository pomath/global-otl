# Function to combine the convolved outputs from LoadDef into a single file.
'''
-Only records the 1st entry in the convolution file.
-Doesn't filter the stations.
'''
import os
import decimal

if __name__ == '__main__':
    conv_dir = '../output/Convolution'
    all_files = os.listdir(path=conv_dir)
    hdr = 'Station,EarthModel,OceanModel,Tide,Lat,Lon,EAmp,EPha,NAmp,NPha,VAmp,VPhas'
    nfiles = len(all_files)
    outstr = [''] * (nfiles + 1)
    outstr[0] = hdr
    for n,f in enumerate(all_files):
        station = f.split('_')[2]
        emodel = f.split('_')[-1].split('.')[0]
        omodel = f.split('_')[-2]
        with open(conv_dir + '/' + f) as fin:
            data = fin.readlines()[1].split()
        tide = data[0].split('-')[1]
        d_array = [decimal.Decimal(x) for x in data[1:]]
        data_str = ','.join([x.to_eng_string() for x in d_array])
        f_string = ','.join([station, emodel, omodel, tide, data_str])
        outstr[n+1] = f_string
    with open('combined_cn.csv', 'w') as f:
        f.write('\n'.join(outstr))
        
