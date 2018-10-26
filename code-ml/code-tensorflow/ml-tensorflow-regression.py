#
# Python 3.6
#

batch_size = 32
epochs = 1000
model_name = ‘model_proba.h5′

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.DESCR)
print(y_train.DESCR)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

input_shape = x_train.shape

model = Sequential()
model.add(Dense(100, input_dim=input_shape[1], activation=’relu’))
model.add(Dropout(0.3))
model.add(Dense(20, activation=’relu’))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss=’mean_squared_error’, optimizer=’adadelta’, metrics=[‘accuracy’])

earlystopper = EarlyStopping(patience=100, verbose=1)
checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1,
callbacks=[earlystopper, checkpointer])

scoret = model.evaluate(x_train, y_train, verbose=0)
score = model.evaluate(x_test, y_test, verbose=0)

print(‘Train loss:’, scoret[0])
print(‘Train accuracy:’, scoret[1])
print(‘Test loss:’, score[0])
print(‘Test accuracy:’, score[1])