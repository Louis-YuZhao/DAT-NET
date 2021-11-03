config = dict()

config['gpu_num'] = '3'

config['dimz'] = 96
config['dimx'] = 96
config['dimy'] = 96
config['channelNum'] = 1

config['odimz'] = 69
config['odimx'] = 95
config['odimy'] = 79

#--------------------------
labelDef = dict()
labelDef['MSA'] = int(0)
labelDef['IPD'] = int(1)
labelDef['PSP'] = int(2)
labelDef['NC'] = int(3)

config['labelDef'] = labelDef
config['target_names'] = ['MSA', 'IPD', 'PSP', 'NC']