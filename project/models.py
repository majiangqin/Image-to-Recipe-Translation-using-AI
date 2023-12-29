from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from .utils import make_iou

def create_model_with_feature_extractor(
    num_ingredients, feature_extractor, metrics=[],
    learning_rate=0.001, input_dropout=0.3,
    model_file=None
):
    #create our final ingredient predictor model consisting of GAP and a dense layer (to be added after feature extractor)
    ingredient_predictor = Sequential(name="ingredient_predictor")
    ingredient_predictor.add(GlobalAveragePooling2D())
    ingredient_predictor.add(Dropout(input_dropout))

    ingredient_predictor.add(Dense(num_ingredients, activation='sigmoid'))

    # create our complete model using the pretrained feature extractor and our final classifier
    model = Sequential([
        feature_extractor,
        ingredient_predictor
    ])

    #compile the model
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    if model_file is not None:
        ingredient_predictor.set_weights(load_model(model_file).get_weights())

    return model, ingredient_predictor
