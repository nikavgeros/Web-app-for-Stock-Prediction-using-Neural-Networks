# Import the libraries
from utils import *


# Run 
if __name__ == "__main__":

    print('\n****** CREATING PATHS ******')
    create_dir(path=image_path)
    create_dir(path=scaler_path)
    create_dir(path=model_path)

    img_dir = image_path + os.listdir(image_path)[0] + "\\"
    scaler_dir = scaler_path + os.listdir(image_path)[0] + "\\"
    model_dir = model_path + os.listdir(image_path)[0] + "\\"

    for stock in stocks:

        print(f'\n****** SELECTED STOCK: {stock} ******')
        data = load_data(stock=stock)

        print('\n****** SPLITTING THE DATASET TO TRAIN AND TEST ******')
        train, test = splitting_data(data=data, split_date=split_date)

        print('\n****** FEATURE SCALING ******')
        train_scaled, scaler = scale_data(data=train, file=f"{scaler_dir}{stock}_Scaler_1layers.pickle", save=True)

        print('\n****** TRAIN TO SUPERVISED ******')
        X_train, y_train = train_to_supervised(data=train_scaled, start=timesteps, end=len(train))
        
        print('\n****** TEST TO SUPERVISED ******')
        X_test = test_to_supervised(data=data[len(data) - len(test) - timesteps:]['Close'].values, start=timesteps, end=len(test)+timesteps, scaler=scaler)

        print('\n****** BUILDING MODELS ******')
        nn_list = [LSTM, GRU, SimpleRNN]
        for nn in nn_list:
            current_nn = build_one_layer_nn(X=X_train, y=y_train, model=nn, first_layer_units=150, file=f'{model_dir}{stock}_{nn.__name__}_1layers.h5')

            print('/n****** PREDICTIONS ******')
            predictions = make_predictions(data=X_test, model=current_nn, scaler=scaler)

            print('/n****** RMSE ******')
            rmse = rmse_calculate(actual=test, predicted=predictions)

            print('/n****** VISUALIZATION ******')
            visualization(actual=test, predicted=predictions, title=f'{stock} Price Prediction with RMSE {round(rmse,3)}', file=f'{img_dir}{stock}_{nn.__name__}_1layers_{round(rmse,3)}.png')


