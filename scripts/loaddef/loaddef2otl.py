"""
m2 s2 n2 k2 k1 o1 p1 q1 Mf Mm Ssa
$$ AMPLITUDES (m)
$$   RADIAL
$$   TANGENTL    EW
$$   TANGENTL    NS
$$ PHASES (degrees)
$$   RADIAL
$$   TANGENTL    EW
$$   TANGENTL    NS

0Extension/Epoch  1Lat(+N,deg)  2Lon(+E,deg)  3E-Amp(mm)  4E-Pha(deg)  5N-Amp(mm)  6N-Pha(deg)  7V-Amp(mm)  8V-Pha(deg) 
"""
stn="ABCD"
EM="PREM"
TM="FES2014"
loc="cm"
fn="../output/Convolution/cn_OceanOnly_" + stn.lower() + "_"+loc+"_convgf_"+TM+"_"+EM.upper()+".txt"
with open(fn) as f:
    data = f.readlines()
ampr = []
ampe = []
ampn = []
phr = []
phe = []
phn = []
harmonics = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA"]
for h in harmonics:
    row = [x for x in data if h in x]
    if not row:
        row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        row = row[0].split()
    ampr.append('{:0.5f}'.format(float(row[7]) / 1e3)[1:])
    ampe.append('{:0.5f}'.format(float(row[3]) / 1e3)[1:])
    ampn.append('{:0.5f}'.format(float(row[5]) / 1e3)[1:])
    phr.append('{:6.1f}'.format(float(row[8])))
    phe.append('{:6.1f}'.format(float(row[4])))
    phn.append('{:6.1f}'.format(float(row[6])))

with open(stn + "_" + TM +"_" + loc + EM + ".otl", 'w') as f:
    f.write('#' + fn + '\n')
    f.write(stn + '\n')
    f.write('  ' + ' '.join(ampr) + '\n')
    f.write('  ' + ' '.join(ampe) + '\n')
    f.write('  ' + ' '.join(ampn) + '\n')
    f.write('  ' + ' '.join(phr) + '\n')
    f.write('  ' + ' '.join(phe) + '\n')
    f.write('  ' + ' '.join(phn) + '\n')
 
 
 
 
 
 
 
 
 
 
 
 

