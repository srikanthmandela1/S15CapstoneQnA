# S15CapstoneQnA


pip install jupyter-toc-generator==0.1.0

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the generator network
def build_generator():
    # Implementation details of the generator network architecture
    generator = keras.Sequential()
    # Add layers to the generator model
    # ...

    return generator

# Define the discriminator network
def build_discriminator():
    # Implementation details of the discriminator network architecture
    discriminator = keras.Sequential()
    # Add layers to the discriminator model
    # ...

    return discriminator

# Initialize the generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Rest of the code...



import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# Define the generator network
def build_generator():
    # Implementation details of the generator network architecture
    generator = keras.Sequential()
    # Add layers to the generator model
    # ...

    return generator

# Define the discriminator network
def build_discriminator():
    # Implementation details of the discriminator network architecture
    discriminator = keras.Sequential()
    # Add layers to the discriminator model
    # ...

    return discriminator

# Define the loss function for the generator
def generator_loss(disc_generated_output, gen_output, target_image):
    # Implementation of the generator loss function
    # ...

    return gen_loss

# Define the loss function for the discriminator
def discriminator_loss(disc_real_output, disc_generated_output):
    # Implementation of the discriminator loss function
    # ...

    return disc_loss

# Initialize the generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Define the optimizers for the generator and discriminator
generator_optimizer = keras.optimizers.Adam()
discriminator_optimizer = keras.optimizers.Adam()

# Load the image dataset
dataset_path = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
dataset_dir = tf.keras.utils.get_file("cifar-10-python.tar.gz", dataset_path, extract=True)
dataset_dir = os.path.join(os.path.dirname(dataset_dir), "cifar-10-batches-py")

# Define the image data generator
image_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.vgg16.preprocess_input,
    validation_split=0.2,
)

# Load the training and validation data
image_size = (32, 32)
batch_size = 32
train_dataset = image_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    subset="training",
    class_mode="input",  # Set class_mode to "input"
)
val_dataset = image_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    subset="validation",
    class_mode="input",  # Set class_mode to "input"
)

# Define the training loop
@tf.function
def train_step(input_image, target_image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images from the generator
        gen_output = generator(input_image, training=True)

        # Discriminator loss calculation
        disc_real_output = discriminator([input_image, target_image], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # Generator loss calculation
        gen_loss = generator_loss(disc_generated_output, gen_output, target_image)

    # Calculate the gradients
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply the gradients
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i, (input_image, target_image) in enumerate(dataset):
            train_step(input_image, target_image)
            if i % 10 == 0:
                print(f"Batch {i+1}/{len(dataset)}")

# Train the model
epochs = 100
train(train_dataset, epochs)

# Generate example outputs
example_input, example_target = next(val_dataset)
example_output = generator.predict(example_input)

# Visualize example outputs
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(4, 4, i+1)
    plt.imshow(example_input[i] * 0.5 + 0.5)
    plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(4, 4, i+1)
    plt.imshow(example_target[i] * 0.5 + 0.5)
    plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(4, 4, i+1)
    plt.imshow(example_output[i] * 0.5 + 0.5)
    plt.axis("off")
plt.tight_layout()
plt.show()
