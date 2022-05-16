# Inspiration: Keith Galli's Intro to Neural Nets
# https://www.youtube.com/watch?v=aBIGJeHRZLQ

from tensorflow import keras

def build_linear_model(train_df):
	model = keras.Sequential([
		keras.layers.Dense(4, input_shape=(2,), activation='relu'),
		keras.layers.Dense(2, activation='sigmoid')])

	model.compile(optimizer='adam', 
				loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])

	X = np.column_stack((train_df.x0.values, train_df.x1.values))

	model.fit(X, train_df.label.values, batch_size=4, epochs=5)
	return model

if __name__ == '__main__':
	import numpy as np
	import pandas as pd
	# load the training data and train the model
	train_df = pd.read_csv('./datasets/linear/train.csv')
	np.random.shuffle(train_df.values)
	print(train_df.head())
	model = build_linear_model(train_df)

	# load the test data and evaluate the model
	test_df = pd.read_csv('./datasets/linear/test.csv')
	test_x = np.column_stack((test_df.x0.values, test_df.x1.values))
	print("EVALUATION")
	model.evaluate(test_x, test_df.label.values)