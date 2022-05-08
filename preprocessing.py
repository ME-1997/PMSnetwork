import torchvision.transforms as transforms


__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}



#normalize image
def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    #transforms.Compose --> Composes several transforms together
    return transforms.Compose(t_list)







def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = __imagenet_stats
    input_size = 256
    return scale_crop(input_size=input_size,
                        scale_size=scale_size, normalize=normalize)









