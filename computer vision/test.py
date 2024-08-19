import tensorflow as tf
import tensorflow_datasets as tfds 

mnist_train = tfds.load(name="fashion_mnist", split="train") 

#Using assert isinstance(mnist_train, tf.data.Dataset) ensures that the variable mnist_train is indeed a TensorFlow dataset.
assert isinstance(mnist_train, tf.data.Dataset) 
for item in mnist_train.take(1): 
    print(type(item)) 
    print(item.keys()) 
    print(item['image']) 
    print(item['label'])

#to check a dataset info 
mnist_test, info = tfds.load(name="horses_or_humans", with_info="True") 
print(info)

# horse and humans using tfds
data=tfds.load('horses_or_humans',split='train',as_supervised=True)
train_batches=data.shuffle(100).batch(10)

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                               input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25), 
        tf.keras.layers.Dense(1,activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

val_data=tfds.load('horses_or_humans',split='test',as_supervised=True)
val_batch=val_data.batch(32)
steps=int(len(val_batch)/32)
history = model.fit(train_batches, epochs=10,validation_data=val_batch,validation_steps=steps)

