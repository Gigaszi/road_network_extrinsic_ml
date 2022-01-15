import geojson
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import linear_model, svm, neural_network
import statsmodels.api as sm
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse as RMSE



from sklearn import metrics


def fit_model(regressor, X_train, y_train):
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return y_pred


def plot_model_fit(model_name, y_pred, y_test, unit):
    # training data
    plt.figure(figsize=(3, 3))
    # limit = 7.5* 10**3
    plt.scatter(
        y_test,
        y_pred,
        alpha=0.1
    )
    # plt.plot(
    #    [0, limit],
    #    [0, limit],
    #    '--'
    # )
    plt.title(model_name)
    plt.ylabel("prediction")
    plt.xlabel(f"street length extern {unit} (realization)")
    plt.show()
    print('explained deviance:', metrics.explained_variance_score(y_test, y_pred))
    R_square = r2_score(y_test, y_pred)
    print('Coefficient of Determination', R_square)
    mse = mean_squared_error(y_test, y_pred)
    print(mse)

path_to_infile = input("Path to input file: ")
path_to_outfile = input("Path to output file: ")

with open(path_to_infile) as f:
    gj = geojson.load(f)
vals_all = []
pops_all = []
osm_all = []
extern_all = []
urban_all = []
shdi_all = []
for el in gj["features"]:
    if el["properties"]["extern_per_sqkm"] is not None and el["properties"]["ghspop_den"] is not None and el["properties"]["shdi_mean"] is not None:
        pops_all.append(el["properties"]["ghspop_den"])
        osm_all.append(el["properties"]["osm_per_sqkm"])
        extern_all.append(el["properties"]["extern_per_sqkm"])
        urban_all.append(el["properties"]["urban"])
        shdi_all.append(el["properties"]["shdi_mean"])

df = pd.DataFrame(extern_all, columns= ["len_extern"])

df[u"population"] = pops_all
df[u"osm"] = osm_all
df[u"extern"] = extern_all
df[u"urban"] = urban_all
df[u"shdi"] = shdi_all

#X = df[[ "population"]]
#X = df[[ "population", "shdi"]]
#X = df[[ "population", "urban"]]
X = df[[ "population", "urban", "shdi"]]

Y = df["len_extern"]


scaler = StandardScaler()
model = scaler.fit(X)
scaled_data = model.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.6)
print(f"there are {len(X_train)} training samples.")



unit = "per sqkm"

LOWER_ALPHA = 0.2
UPPER_ALPHA = 0.8

models = [
    #["random_forest", RandomForestRegressor(n_estimators=100)],
    ["OLM", linear_model.LinearRegression()],
   # ["lower", GradientBoostingRegressor(loss="quantile", alpha=LOWER_ALPHA)],
    #["middle", GradientBoostingRegressor(loss="squared_error")],
    #["upper", GradientBoostingRegressor(loss="quantile", alpha=UPPER_ALPHA)],
   # ["GLM", linear_model.PoissonRegressor(alpha=1e-12, max_iter=300)],
    # ["SVM", svm.SVR()],
    # ["MLP", neural_network.MLPRegressor(random_state=1, max_iter=5000)]
]
te = y_test.array.reshape(1, -1)


for model_name, regressor in models:
    y_pred = fit_model(regressor, X_train, y_train)
    plot_model_fit(model_name, y_pred, y_test, unit)
    print(f"trained model: {model_name}")

for model_name, regressor in models:
    y_pred = regressor.predict(X)
    df[f"prediction_{model_name}"] = y_pred

    df[f"osm_difference_{model_name}"] = df[f"prediction_{model_name}"] - df[f"osm"]
    df[f"osm_completness_{model_name}"] = round(df[f"osm"] / df[f"prediction_{model_name}"],
                                                    3)
    df[f"extern_difference_{model_name}"] = df[f"prediction_{model_name}"] - df[f"extern"]
    df[f"extern_completness_{model_name}"] = round(df[f"extern"] / df[f"prediction_{model_name}"],
                                    3)
    # plot_prediction(model_name, y_pred, y, unit)
    print(f"predicted model: {model_name}")
    for index, row in df.iterrows():
        gj["features"][index]["properties"].update({f"prediction_{model_name}": row[f"prediction_{model_name}"]})
        gj["features"][index]["properties"].update({f"extern_completness_{model_name}": row[f"extern_completness_{model_name}"]})
        gj["features"][index]["properties"].update({f"osm_completness_{model_name}": row[f"osm_completness_{model_name}"]})


with open(path_to_outfile, 'w') as f2:
  geojson.dump(gj, f2)


