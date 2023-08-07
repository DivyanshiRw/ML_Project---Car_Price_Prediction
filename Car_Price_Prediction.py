
# # FAIR CAR PRICE PREDICTION
# 


import pandas as pd



cars=pd.read_csv("cars24_usedCars.csv")



cars.head()




cars.info()




cars.describe()




#get_ipython().run_line_magic('matplotlib', 'inline')




# import matplotlib.pyplot as plt
# cars.hist(bins=50, figsize=(20,15))


# # Train-Test Split



from sklearn.model_selection import train_test_split
train_set, test_set=train_test_split(cars, test_size=0.2, random_state=42)
# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")




cars['Owner'].value_counts()





cars['Drive'].value_counts()





from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(cars, cars['Drive']):
    strat_train_set=cars.loc[train_index]
    strat_test_set=cars.loc[test_index]

    
strat_train_set['Drive'].value_counts()
strat_test_set['Drive'].value_counts()




#Before looking for correlations, make sure to create a copy of the data
cars=strat_train_set.copy()


# # Looking for correlations



cars_temp=cars.copy()
columns_to_exclude = ['Car Name', 'Fuel', 'Location', 'Drive', 'Type']  # Add the column names you want to exclude

# Drop the specified columns from the temporary dataset
cars_temp.drop(columns=columns_to_exclude, inplace=True)




corr_matrix=cars_temp.corr()
corr_matrix['Price'].sort_values(ascending=False)



# from pandas.plotting import scatter_matrix
# attributes=['Price','Year','Owner','Distance']
# scatter_matrix(cars_temp[attributes], figsize=(12,8))



#cars.plot(kind='scatter', x='Distance', y='Price', alpha=0.8)



cars=strat_train_set.drop("Location", axis=1, inplace=True)
cars=strat_test_set.drop("Location", axis=1, inplace=True)

cars=strat_train_set.drop("Unnamed: 0", axis=1, inplace=True)
cars=strat_test_set.drop("Unnamed: 0", axis=1, inplace=True)



cars=strat_train_set.drop("Price", axis=1)
cars_labels=strat_train_set['Price'].copy()






cars.shape





cars.info()





median= cars['Year'].median()           
cars['Year'].fillna(median, inplace=True)





mode_car_name = cars['Car Name'].mode().iloc[0]
cars['Car Name'].fillna(mode_car_name, inplace=True)





#To deal with missing values which might be encountered in future
from sklearn.impute import SimpleImputer

# Create separate DataFrames for numeric and non-numeric columns
numeric_columns = cars.select_dtypes(include='number')
non_numeric_columns = cars.select_dtypes(exclude='number')

# Apply imputation to numeric columns
numeric_imputer = SimpleImputer(strategy='median')
imputed_numeric_columns = pd.DataFrame(numeric_imputer.fit_transform(numeric_columns), columns=numeric_columns.columns)

# Apply imputation to string columns
non_numeric_imputer = SimpleImputer(strategy='most_frequent')
imputed_non_numeric_columns = pd.DataFrame(non_numeric_imputer.fit_transform(non_numeric_columns), columns=non_numeric_columns.columns)


# Combine the imputed numeric and non-numeric columns back into the final DataFrame
cars_imputed= pd.concat([imputed_numeric_columns, imputed_non_numeric_columns], axis=1)






cars_imputed.describe()




#cars= cars_imputed.copy()




cars_imputed.info()




cars.info()


# # Creating a Pipeline



from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



# Create separate DataFrames for numeric and non-numeric columns
numeric_columns = cars.select_dtypes(include='number')
non_numeric_columns = cars.select_dtypes(exclude='number')

# Create a pipeline for numeric columns
numeric_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

# Create a pipeline for non-numeric (string) columns
non_numeric_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the column transformer
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_columns.columns),
    ('non_numeric', non_numeric_pipeline, non_numeric_columns.columns)
])

# Fit and transform the data using the preprocessor pipeline




cars_transformed = preprocessor.fit_transform(cars_imputed)


# # Selecting a desired model
# 



from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.svm import SVR

model=LinearRegression()
#model= DecisionTreeRegressor()
# model= RandomForestRegressor()
#model=GradientBoostingRegressor()
#model= SVR()
model.fit(cars_transformed, cars_labels)




some_data=cars.iloc[:5]
some_labels=cars_labels.iloc[:5]




prepared_data=preprocessor.transform(some_data)
model.predict(prepared_data)




list(some_labels)


# # Evaluating the model



import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import metrics
cars_predictions=model.predict(cars_transformed)
mse=mean_squared_error(cars_labels, cars_predictions)
rmse=np.sqrt(mse)




rmse


# # Cross Validation



from sklearn.model_selection import cross_val_score
scores=cross_val_score(model, cars_transformed, cars_labels, scoring='neg_mean_squared_error', cv=10)
rmse_scores=np.sqrt(-scores)




rmse_scores




def print_scores(scores):
    print('scores: ', scores)
    print('mean: ', scores.mean())
    print('standard deviation: ', scores.std())
    




# print_scores(rmse_scores)




from sklearn.metrics import r2_score

# Make predictions on the training data
cars_predictions = model.predict(cars_transformed)

# Calculate the R-squared score for the training data
r2_train = r2_score(cars_labels, cars_predictions)

# print(f"R-squared score on the training data: {r2_train*100:.2f}%")


# # Saving the model



from joblib import dump, load
dump(model, 'DR_carPrice.joblib')


# # Testing the model




X_test = strat_test_set.drop("Price", axis=1)
Y_test = strat_test_set["Price"].copy()

X_test_prepared = preprocessor.transform(X_test)
final_prediction = model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test, final_prediction)
final_rmse=np.sqrt(final_mse)

final_rmse




final_r_squared = model.score(X_test_prepared, Y_test)
# print(f"R-squared score on the test set: {final_r_squared*100:.2f}%")




# print(final_prediction, list(Y_test))




prepared_data_array = prepared_data[0].toarray()
# print(prepared_data_array)






# # Creating Function to use the model



cars['Type'].unique()




cars['Drive'].unique()


cars['Fuel'].unique()




def get_user_input():
    car_name = input("Enter the car name: ").lower()
    year = float(input("Enter the car year: "))
    distance = int(input("Enter the car distance: "))
    owner = int(input("Enter the number of previous owners: "))
    fuel = input("Enter the fuel type (PETROL /DIESEL /CNG /LPG): ").lower()
    drive = input("Enter the drive type (Manual/ Automatic): ").lower()
    car_type = input("Enter the car type (HatchBack /Sedan /SUV /Lux_SUV/ Lux_sedan): ").lower()

    return car_name, year, distance, owner, fuel, drive, car_type




def preprocess_user_input(user_input):
    car_name, year, distance, owner, fuel, drive, car_type = user_input

    # Create a DataFrame with the user input
    user_data = pd.DataFrame({
        'Car Name': [car_name],
        'Year': [year],
        'Distance': [distance],
        'Owner': [owner],
        'Fuel': [fuel],
        'Drive': [drive],
        'Type': [car_type]
    })

    # Apply imputation and other preprocessing steps
    numeric_columns = user_data.select_dtypes(include='number')
    numeric_imputer = SimpleImputer(strategy='median')
    imputed_numeric_columns = pd.DataFrame(numeric_imputer.fit_transform(numeric_columns), columns=numeric_columns.columns)

    non_numeric_columns = user_data.select_dtypes(exclude='number')
    non_numeric_imputer = SimpleImputer(strategy='most_frequent')
    imputed_non_numeric_columns = pd.DataFrame(non_numeric_imputer.fit_transform(non_numeric_columns), columns=non_numeric_columns.columns)

    user_data_imputed = pd.concat([imputed_numeric_columns, imputed_non_numeric_columns], axis=1)

    # Apply one-hot encoding to categorical features
    user_data_transformed = preprocessor.transform(user_data_imputed)

    return user_data_transformed




def predict_car_price(user_input_transformed):
    price_prediction = model.predict(user_input_transformed)
    return price_prediction[0]




# Using the model

def main():
    user_input = get_user_input()
    user_input_transformed = preprocess_user_input(user_input)
    predicted_price = predict_car_price(user_input_transformed)
    print(f"Predicted Car Price: {predicted_price:.2f}")

if __name__ == "__main__":
    main()

