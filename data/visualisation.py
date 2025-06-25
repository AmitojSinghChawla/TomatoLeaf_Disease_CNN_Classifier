from data_loading import dataset
import matplotlib.pyplot as plt
#Show the first 10 images in the dataset
import random
for i in random.sample(range(len(dataset)), 10):
    image,label = dataset[i]
    class_name=dataset.classes[label]
    image=image.permute(1,2,0).numpy() # as image tensors are in channel first format(channels,height,width) and matplotlib.imshow expects them as channel last format using permute we are converting the tensor to (height,width,channels) format.
    if image.max() <= 1.0:
        image = (image * 255).astype('uint8')

#Some image viewers and imshow() may show slightly dim or washed-out colors if the image stays in [0.0–1.0] float format.

#Converting to [0–255] uint8 gives correct color fidelity across all environments.

    plt.imshow(image)
    plt.title(class_name)
    plt.axis('off')
    plt.show()

