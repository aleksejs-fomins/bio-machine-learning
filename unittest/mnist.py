from lib.data import mnist

mnistData = mnist.load("/home/aleksejs/Downloads/mnist_data/")

select_images = [0,1, 15, 138, 2000]

mnist.plot(mnistData["train_images"], mnistData["train_labels"], select_images)