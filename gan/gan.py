import json
import os
import time

import numpy as np
import numpy.random
import tensorflow as tf
from tensorflow.keras import layers, Model

import data_images
import gan_excel
from utils import file_management

MIN_RECT_DIMS = (4, 4)
MAX_RECT_DIMS = (4, 16)
# -- GENERATOR_TYPES --
# 0: No dims;
# 1: Single dimension given as input and the other is fixed;
# 2: Two dims given in a single input;
# 3: Two dims given in separate inputs.
GENERATOR_TYPE = 0
IMAGE_DIMS = (16, 20)
NOISE_AS_RECT = True  # Format noise pixels to have corresponding cells dimensions
data_images.MINIMUM_SAMPLE_SIZE_FOR_EACH_DIMENSIONS = 0
BATCH_SIZE = 32
SEMANTIC_BATCH_SIZE = 4
# -- SEMANTIC_LOSS_VERSIONS --
# 0: No semantic loss;
# 1: Comparison between real and guessed rectangle dimensions;
# 2: Proportion of each cell that is not the dominant color.
SEMANTIC_LOSS_VERSION = 2


def make_generator_model():
    input_noise = layers.Input(shape=(IMAGE_DIMS[0] * IMAGE_DIMS[1]))

    if GENERATOR_TYPE == 0:
        x = input_noise
    elif GENERATOR_TYPE == 1:
        input_rect_h = layers.Input(shape=(1,))
        x = layers.concatenate([input_noise, input_rect_h])
    elif GENERATOR_TYPE == 2:
        input_rect = layers.Input(shape=(2,))
        x = layers.concatenate([input_noise, input_rect])
    elif GENERATOR_TYPE == 3:
        input_rect_b = layers.Input(shape=(1,))
        input_rect_h = layers.Input(shape=(1,))
        x = layers.concatenate([input_noise, input_rect_b, input_rect_h])
    else:
        raise RuntimeError('Invalid generator type')

    x = layers.Dense(IMAGE_DIMS[0] * IMAGE_DIMS[1] * 2, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((IMAGE_DIMS[0], IMAGE_DIMS[1], 2))(x)

    x = layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    output = layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False,
                                    activation='softmax')(
        x)

    if GENERATOR_TYPE == 0:
        return Model(inputs=input_noise, outputs=output)
    elif GENERATOR_TYPE == 1:
        return Model(inputs=[input_noise, input_rect_h], outputs=output)
    elif GENERATOR_TYPE == 2:
        return Model(inputs=[input_noise, input_rect], outputs=output)
    elif GENERATOR_TYPE == 3:
        return Model(inputs=[input_noise, input_rect_b, input_rect_h], outputs=output)


def make_discriminator_model():
    input_image = layers.Input(shape=(IMAGE_DIMS[0], IMAGE_DIMS[1], 256))
    input_rect_dims = layers.Input(shape=(2,))

    x = layers.Dense(IMAGE_DIMS[0] * IMAGE_DIMS[1], use_bias=False)(input_rect_dims)
    x = layers.Reshape((IMAGE_DIMS[0], IMAGE_DIMS[1], 1))(x)
    x = layers.concatenate([input_image, x])

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    return Model(inputs=[input_image, input_rect_dims], outputs=output)


class GAN:
    def __init__(self):
        self.generator = make_generator_model()
        self.discriminator = make_discriminator_model()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        self.model_dir = None
        self.checkpoint_manager = None
        self.config_number = None
        self.after_config_update()

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def _semantic_loss(self, generated_images, rect_dims):
        if SEMANTIC_LOSS_VERSION == 1:  # Version 1: Uses comparison between real and guessed rectangle dimensions
            def mapping_func(image_array):
                image = data_images.one_hots_to_array_img(image_array)
                return data_images.detect_cell_dims(image)

            outputs = [mapping_func(image_array) for image_array in generated_images.numpy()[-SEMANTIC_BATCH_SIZE:]]
            return self.cross_entropy(tf.cast(outputs, tf.float32),
                                      tf.cast(rect_dims[-SEMANTIC_BATCH_SIZE:], tf.float32))

        elif SEMANTIC_LOSS_VERSION == 2:  # Version 2: Uses the proportion of each cell that is not the dominant color
            def mapping_func(image_array, dims):
                image = data_images.one_hots_to_array_img(image_array)
                return data_images.mean_cell_standard_deviation(image, dims)

            outputs = [mapping_func(image_array, dims) for image_array, dims in
                       zip(generated_images.numpy()[-SEMANTIC_BATCH_SIZE:], rect_dims[-SEMANTIC_BATCH_SIZE:])]
            return np.mean(outputs)

        else:  # Semantic loss won't be applied
            return 0

    def after_config_update(self):
        config = {'MIN_RECT_DIMS': list(MIN_RECT_DIMS),
                  'MAX_RECT_DIMS': list(MAX_RECT_DIMS),
                  'IMAGE_DIMS': list(IMAGE_DIMS),
                  'NOISE_AS_RECT': NOISE_AS_RECT,
                  'MINIMUM_SAMPLE_SIZE_FOR_EACH_DIMENSIONS': data_images.MINIMUM_SAMPLE_SIZE_FOR_EACH_DIMENSIONS,
                  'BATCH_SIZE': BATCH_SIZE,
                  'SEMANTIC_BATCH_SIZE': SEMANTIC_BATCH_SIZE}

        config_file_path = 'configs.json'
        self.config_number = None
        with open(config_file_path) as json_file:
            configs = json.load(json_file)
            for k, c in configs.items():
                if config == c:
                    self.config_number = k
        if self.config_number is None:
            self.config_number = len(configs)
            configs[self.config_number] = config
            with open(config_file_path, 'w') as json_file:
                json.dump(configs, json_file)
            print(f'Added config {self.config_number} to configs.json')
        else:
            print(f'Current config number is {self.config_number}')

        self.model_dir = f'trained/{self.config_number}/gen{GENERATOR_TYPE}_sem{SEMANTIC_LOSS_VERSION}'
        if os.path.exists(self.model_dir) is False:
            os.makedirs(f'{self.model_dir}/validation')
            os.makedirs(f'{self.model_dir}/checkpoints')
            os.makedirs(f'{self.model_dir}/test')

        checkpoint_dir = f'./{self.model_dir}/checkpoints'
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=1)

    def predict(self, rect_dims):
        if NOISE_AS_RECT or GENERATOR_TYPE == 1 or GENERATOR_TYPE == 3 or SEMANTIC_LOSS_VERSION != 0:
            tf.config.run_functions_eagerly(True)

        if NOISE_AS_RECT:
            noise = np.reshape([data_images.generate_noise_with_image_dims(rect_dims, IMAGE_DIMS)],
                               (1, IMAGE_DIMS[0] * IMAGE_DIMS[1]))
        else:
            noise = tf.random.normal([1, IMAGE_DIMS[0] * IMAGE_DIMS[1]])

        if GENERATOR_TYPE == 0:
            predictions = self.generator(noise, training=False)
        elif GENERATOR_TYPE == 1:
            predictions = self.generator([noise, np.array([rect_dims[1]])], training=False)
        elif GENERATOR_TYPE == 2:
            predictions = self.generator([noise, np.array([rect_dims])], training=False)
        elif GENERATOR_TYPE == 3:
            predictions = self.generator([noise, np.array([rect_dims[0]]), np.array([rect_dims[1]])], training=False)
        else:
            raise RuntimeError('Invalid generator type')
        return predictions

    def restore(self):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def test(self, rect_dims):
        file_management.empty_directory(f'{self.model_dir}/test/')
        b, h = rect_dims
        real_snakes = np.load(f"../data/npy/snake_{b}x{h}.npy", allow_pickle=True)

        success = 0
        number_of_test = 100
        for i in range(number_of_test):
            predictions = self.predict(rect_dims)

            img = data_images.one_hots_to_image(predictions[0].numpy())
            fake_snake_bh = data_images.image_to_snake(img, (b, h))
            fake_snake_hb = data_images.image_to_snake(img, (h, b))

            saved = False
            for real_snake in real_snakes:
                if fake_snake_bh == real_snake or fake_snake_hb == real_snake:
                    img.save(f'{self.model_dir}/test/+{i}_{b}x{h}.png')
                    file_management.delete_file(
                        f'{self.model_dir}/test/{i}_{b}x{h}.png')
                    success += 1
                    break
                elif not saved:
                    img.save(f'{self.model_dir}/test/{i}_{b}x{h}.png')
                    saved = True

        print(f"\nSuccess rate: {success * 100 / number_of_test}%")

    def train(self, dataset, epochs, squares_only=False, validation_xlsx=False):
        if NOISE_AS_RECT or GENERATOR_TYPE == 1 or GENERATOR_TYPE == 3 or SEMANTIC_LOSS_VERSION != 0:
            tf.config.run_functions_eagerly(True)

        file_management.empty_directory(f'{self.model_dir}/validation/')
        rng = numpy.random.default_rng()
        excess = dataset.shape[1] % BATCH_SIZE
        number_of_batches = dataset.shape[1] // BATCH_SIZE
        validation_rates = {}
        for epoch in range(epochs):
            start = time.time()
            rng.shuffle(dataset, axis=1)
            real_all = np.stack(dataset[0])
            rect_dims_all = np.array(dataset[1].tolist())
            if excess != 0:
                real_all = real_all[:-excess]
                rect_dims_all = rect_dims_all[:-excess]
            real_batches = np.split(real_all, number_of_batches)
            rect_dims_all = np.sort(rect_dims_all, axis=1)
            rect_dims_batches = np.split(rect_dims_all, number_of_batches)
            for real_batch, rect_dims_batch in zip(real_batches, rect_dims_batches):
                self._train_step(real_batch, rect_dims_batch)
            print(f"\rTime for epoch {epoch + 1}/{EPOCHS} is {time.time() - start} sec", end="")
            # Save the model every 250 epochs
            if (epoch + 1) % 25 == 0:
                self.checkpoint_manager.save()
                validation_rates[epoch + 1] = self.validate(epoch + 1, squares_only=squares_only)
        if validation_xlsx:
            gan_excel.export_validation_rates(validation_rates,
                                              f'{self.config_number}-{GENERATOR_TYPE}-{SEMANTIC_LOSS_VERSION}')
            print(f'Exported validation rates as .xlsx')

    @tf.function
    def _train_step(self, images, rect_dims):
        if NOISE_AS_RECT is True:
            noises = np.array([data_images.generate_noise_with_image_dims(dims, IMAGE_DIMS) for dims in rect_dims])
            noises = np.reshape(noises, (BATCH_SIZE, IMAGE_DIMS[0] * IMAGE_DIMS[1]))
        else:
            noises = tf.random.normal([len(rect_dims), IMAGE_DIMS[0] * IMAGE_DIMS[1]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            if GENERATOR_TYPE == 0:
                generated_images = self.generator([noises], training=True)
            elif GENERATOR_TYPE == 1:
                generated_images = self.generator([noises, np.array([dims[1] for dims in rect_dims])], training=True)
            elif GENERATOR_TYPE == 2:
                generated_images = self.generator([noises, rect_dims], training=True)
            elif GENERATOR_TYPE == 3:
                generated_images = self.generator([noises,
                                                   np.array([dims[0] for dims in rect_dims]),
                                                   np.array([dims[1] for dims in rect_dims])], training=True)
            else:
                raise RuntimeError('Invalid generator type')

            real_output = self.discriminator([images, rect_dims], training=True)
            fake_output = self.discriminator([generated_images, rect_dims], training=True)

            sem_loss = self._semantic_loss(generated_images, rect_dims)
            gen_loss = self._generator_loss(fake_output) + sem_loss
            disc_loss = self._discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def validate(self, epoch, squares_only=False) -> float:
        success = 0
        number_of_test = 1000
        for i in range(number_of_test):
            b = MAX_RECT_DIMS[0]
            h = np.random.randint(MIN_RECT_DIMS[1], MAX_RECT_DIMS[1] + 1)
            dims = (b, h)
            predictions = self.predict(dims)

            img = data_images.one_hots_to_image(predictions[0].numpy())
            fake_snake_bh = data_images.image_to_snake(img, (b, h))
            fake_snake_hb = data_images.image_to_snake(img, (h, b))

            real_snakes = np.load(f"../data/npy/snake_{dims[0]}x{dims[1]}.npy", allow_pickle=True)
            saved = False
            for real_snake in real_snakes:
                if b > h:
                    real_snake.rotate_grid()
                if fake_snake_bh == real_snake or fake_snake_hb == real_snake:
                    img.save(f'{self.model_dir}/validation/+{epoch}_{i}_{b}x{h}.png')
                    file_management.delete_file(
                        f'{self.model_dir}/validation/{epoch}_{i}_{b}x{h}.png')
                    success += 1
                    break
                elif not saved:
                    img.save(f'{self.model_dir}/validation/{epoch}_{i}_{b}x{h}.png')
                    saved = True
        validation_rate = success * 100 / number_of_test
        print(f"\nValidation rate: {validation_rate}%")
        return validation_rate


if __name__ == '__main__':
    EPOCHS = 2000
    squares_only = False
    if squares_only:
        train_dataset = data_images.load_square_dims_data(MIN_RECT_DIMS[0], MAX_RECT_DIMS[0], IMAGE_DIMS, "MAX_SNAKES")
    else:
        train_dataset = data_images.load_varied_dims_data(MIN_RECT_DIMS, MAX_RECT_DIMS, IMAGE_DIMS, "MAX_SNAKES")

    # TRAIN SINGLE MODEL
    np.random.seed(0)
    gan = GAN()
    gan.train(train_dataset, EPOCHS, squares_only=squares_only, validation_xlsx=True)

    # TRAIN MULTIPLE MODELS
    # for gen_type in range(4):
    #     for sem_type in range(0, 3):
    #         np.random.seed(0)
    #         GENERATOR_TYPE = gen_type
    #         SEMANTIC_LOSS_VERSION = sem_type
    #         gan = GAN()
    #         gan.train(train_dataset, EPOCHS, squares_only=squares_only, validation_xlsx=True)
    #         # gan.test(MAX_RECT_DIMS)

    # TEST IF GENERATOR GOES BEYOND TRAINING DIMS
    # gan.restore()
    # gan.test((4, 20))
