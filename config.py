class Config:
    dev = "cuda:0"

    # Random crops to take from the image during training and data augmentation.
    # This value should be large enough compared to your image size (in this case image size was (200 X 300) pixels.
    crop = [100, 100]

    # This should reflect the mean and standard deviation in your dataset (or leave default for 8 bit images)
    sub = 127
    div = 256

    # Maximum allowed voting vector length. All votes are divided by this value during training.
    # This value should roughly be twice the max diameter of an object.
    max_dist = 40

    # Number of epochs to train. The best model with the best validation loss is kept automatically.
    # This value should typically be large. Check vectors.npy if the quality of vectors is ok.
    epochs = 1000

    # Batch size for training. Choose a batch size which maximizes GPU memory usage.
    batch_size = 20

    # Learning rate. Usually this value is sufficient.
    learn_rate = 0.001

    # Determines on what interval validation should occur.
    validation_interval = 10

    # Number of input channels of the image. Default is RGB.
    num_channels = 3

    # Number of classes. This is including the background, so this should be one more than in the training file.
    num_classes = 4

    # How far apart should two centroids minimally be. (values could be: 3, 7, 11, 15, 17, etc.)
    # Increase value to get less detection close together.
    # Decrease value to improve detection which are very close together.
    nm_size = 7

    # How many votes constitutes a centroid.
    # Increase value to increase precision and decrease recall (less detections)
    # Decrease value to increase recall and decrease precision (more detections)
    # Determine a correct value by reviewing centroid_votes.npy
    centroid_threshold = 30

    # How many votes constitutes a border.
    # Determine a correct value by reviewing border_votes.npy
    border_threshold = 1
