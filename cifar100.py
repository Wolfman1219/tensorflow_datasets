import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Input, MaxPooling2D, GlobalAveragePooling2D, Dense

# Define ResNet-9 Architecture

def resnet9_block(x, filters):
    res = Conv2D(filters, (3, 3), padding='same')(x)
    res = BatchNormalization()(res)
    res = ReLU()(res)
    res = Conv2D(filters, (3, 3), padding='same')(res)
    res = BatchNormalization()(res)

    shortcut = Conv2D(filters, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    res = Add()([shortcut, res])
    res = ReLU()(res)

    return res

def resnet9(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = resnet9_block(x, 64)
    x = resnet9_block(x, 128)
    x = resnet9_block(x, 256)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

# Load CIFAR-100 dataset

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1

train_images, test_images = train_images / 255.0, test_images / 255.0

# Compile and train the model

model = resnet9((32, 32, 3), 100)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluate the model

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')
model.save("stream_model/cifar100.h5")
