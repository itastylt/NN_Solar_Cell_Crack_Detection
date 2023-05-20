from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
from skimage import io, img_as_ubyte
import skimage.transform as trans
from PIL import Image

PATH_TO_IMAGES = 'images'
PATH_TO_LABELS = 'masks'
TEST_IMAGE_COUNT = 31


def Unet(input_size = (256, 256, 1), filters = 64):
    inputs = Input(input_size)

    #Encoder    
    conv1 = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=filters*2, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(filters=filters*2, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=filters*4, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(filters=filters*4, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters=filters*8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(filters=filters*8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    #Middle
    conv5 = Conv2D(filters=filters*16, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(filters=filters*16, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    #Decoder
    upconv6 = Conv2D(filters=filters*8, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    merge6 = concatenate([drop4, upconv6], axis=3)
    conv6 = Conv2D(filters=filters*8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(filters=filters*8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    upconv7 = Conv2D(filters=filters*4, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, upconv7], axis=3)
    conv7 = Conv2D(filters=filters*4, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(filters=filters*4, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    upconv8 = Conv2D(filters=filters*2, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, upconv8], axis=3)
    conv8 = Conv2D(filters=filters*2, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(filters=filters*2, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    upconv9 = Conv2D(filters=filters, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, upconv9], axis=3)
    conv9 = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv9 = Conv2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def normalizeData(image, mask, classes = 2):
    if(np.max(image) > 1):
        image = image / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    
    return (image, mask)

def trainDataGenerator():
    imageDataGenerator = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    maskDataGenerator = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    imageGenerator = imageDataGenerator.flow_from_directory(
        directory='train',
        classes=[PATH_TO_IMAGES],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir='datagen/images',
        save_prefix='cell_gen',
        seed=1
    )

    maskGenerator = maskDataGenerator.flow_from_directory(
        directory='train',
        classes=[PATH_TO_LABELS],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=2,
        save_to_dir='datagen/masks',
        save_prefix='mask_gen',
        seed=1
    )

    train_generator = zip(imageGenerator, maskGenerator)

    for (img, mask) in train_generator:
        img, mask = normalizeData(img, mask)
        yield (img, mask)

def testDataGenerator(imageCount = 30):
    for i in range(imageCount):
        image = io.imread(f"test/test{i}.png", as_gray=True)
        image = image / 255
        image = trans.resize(image, (256, 256))
        image = np.reshape(image,image.shape+(1,))
        image = np.reshape(image,(1,)+image.shape)
        yield image

def saveResults(photos):
    for i, item in enumerate(photos):
        image = img_as_ubyte(item[:, :, 0])
        io.imsave(f"results/predict_{i}.png", image)

def main():

    trainData = trainDataGenerator()

    model = Unet()
    model_checkpoint = ModelCheckpoint('unetCells.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(trainData, steps_per_epoch=300, epochs=10, callbacks=[model_checkpoint])

    testData = testDataGenerator(imageCount=TEST_IMAGE_COUNT)

    # If you just want to predict then comment model.fit() and uncomment model.load_weights()
    # model.load_weights("unetCells.hdf5")

    results = model.predict_generator(testData,TEST_IMAGE_COUNT,verbose=1)

    saveResults(results)

    return


if __name__ == "__main__":
    main()