from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from src.utils.train_utils import post_process


class MathewTrainer:
    def __init__(self):
        self.image_x = 100
        self.image_y = 100
        self.train_dir = "data/"
        self.batch_size = 64
        self.model_name = "model/mathew.h5"

    def keras_model(self, image_x, image_y):
        num_of_classes = 14
        model = Sequential()
        model.add(Conv2D(32, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Conv2D(64, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Conv2D(128, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(num_of_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        filepath = self.model_name
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        return model, callbacks_list

    def train(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            rotation_range=15,
            zoom_range=0.2,
            horizontal_flip=False,
            validation_split=0.2,
            fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.image_x, self.image_y),
            color_mode="grayscale",
            batch_size=self.batch_size,
            seed=42,
            class_mode='categorical',
            subset="training",
            shuffle=True)

        validation_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.image_x, self.image_y),
            color_mode="grayscale",
            batch_size=self.batch_size,
            seed=42,
            class_mode='categorical',
            subset="validation",
            shuffle=False)
        print(validation_generator.class_indices)
        model, callbacks_list = self.keras_model(self.image_x, self.image_y)
        print(model.summary())
        his = model.fit_generator(train_generator, epochs=20, validation_data=validation_generator)
        model.save(self.model_name)
        post_process(model, validation_generator, his)
