import matplotlib
matplotlib.use("Qt5Agg")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def show_subplots(fig_list=None, labels=None, direction='h'):
    """
    Function plots the 1 or 3-channel figures using subplots
    @param labels:
    @param fig_list:
    @param direction: h or v
    """

    if fig_list == None:
        return

    num_plots = len(fig_list)
    if direction == 'v':
        fig, axs = plt.subplots(num_plots)
    else:
        fig, axs = plt.subplots(1, num_plots)

    fig.suptitle('Vertically stacked range images')
    for i in range(num_plots):
        axs[i].imshow(fig_list[i])  # depth channel of the 3 channel normalized range image

        if labels is not None:
            axs[i].set_title(labels[i])
    plt.show()
    return fig

def train_mnist():

    print(f"tensorflow version -> {tf.__version__}")
    print(f"Devices:\n {tf.config.list_physical_devices()}")

    # load built in dataset using keras, normalize [0, 1]
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print(f"dataset shape, x_train-> {x_train.shape}, Y-train -> {y_train.shape}")

    # plot 4 samples
    sample_0, sample_0_label = x_train[0], y_train[0]
    sample_1, sample_1_label = x_train[5000], y_train[5000]
    sample_2, sample_2_label = x_train[1500], y_train[1500]
    sample_3, sample_3_label = x_train[50000], y_train[50000]
    # show_subplots([sample_0, sample_1, sample_2, sample_3], 
    #               [str(sample_0_label), str(sample_1_label), str(sample_2_label), str(sample_3_label)])
    
    print("Done")

    # create a  sequential model - only for single input-output layers and models
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # define a loss term
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # the model outputs logits from final layer

    # compile the model
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    # start training
    model.fit(x_train, y_train, batch_size=128, epochs=10)

    # evaluate
    model.evaluate(x_test,  y_test, verbose=2)

    # predict
    preds = model.predict(x_test[1000: 1004, :, :])

    preds_labels = tf.argmax(tf.keras.layers.Softmax()(preds), axis=-1).numpy()
    print(preds_labels.shape)
    print(type(preds_labels))

    show_subplots([x_test[1000], x_test[1001], x_test[1002], x_test[1003]],
                  [str(y_test[1000])+'/'+str(preds_labels[0]),
                   str(y_test[1001])+'/'+str(preds_labels[1]),
                   str(y_test[1002])+'/'+str(preds_labels[2]),
                   str(y_test[1003])+'/'+str(preds_labels[3])])
    
    print("Done")











if __name__ == "__main__":

    train_mnist()

