fp = open("loss_log.txt",'r')
out = open("parsed_out.txt","w")

skiplines=1
iternum = 1000
lines = fp.readlines()
lines= lines[skiplines:]

def func(line):
    line = line.strip()
    ele= line.split(':')
    temp=[]
    for i,e in enumerate(ele):
        temp+=e.split(',')
    temp2 = []
    for i,e in enumerate(temp):
        temp2+=e.split(')')
    final =[]
    for i,e in enumerate(temp2):
        final+=e.split(' ')
    final = [i for i in final if i!='']
    print(final)
    ep, it,kld ,gan, gan_ft, vgg, dfake, dreal = final[1],final[3],final[7],final[9],final[11],final[13],final[15],final[17]
    return ep,it,kld,gan,gan_ft,vgg,dfake,dreal

epoch = []
GAN = []
out.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format('epoch','KLD' ,'GAN', "GAN_FT", "VGG", "D_fake", "D_real"))
for l in lines:
    if l[0]=='=':
        continue
    ep,it,kld,gan,gan_ft,vgg,dfake,dreal = func(l)

    if int(it)==iternum:
        out.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(ep,kld ,gan, gan_ft, vgg, dfake, dreal))


fp.close()
out.close()
