import torch

def generate_images(inputs, outputs, n):
    
    inputs = inputs[:n]
    outputs = outputs[:n]
            
    nchan = inputs.size()[1]
    imsize= inputs.size()[2]
        
    comparison = torch.zeros(nchan, 2*imsize, 15*imsize)
        
    for i in range(0,n):
        step = i*imsize
        comparison[:,:imsize,step:step+imsize] = inputs[i]
        comparison[:,imsize:,step:step+imsize] = outputs[i]
            
    return comparison