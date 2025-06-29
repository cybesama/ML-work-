import tensorflow as tf
data=tf.keras.datasets.fashion_mnist
(training_images,training_label),(test_images,test_label)=data.load_data()
training_images  = training_images / 255.0
test_images = test_images / 255.0
model=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                  tf.keras.layers.Dense(128,activation=tf.nn.relu),
                                  tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images,training_label, epochs=6)

classify=model.predict(test_images)
for i in range(2):
    print(classify[i])
    print(test_label[i])

