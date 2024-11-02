import dml_estimators

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

regressor = RandomForestRegressor(n_estimators=300, max_depth=7, max_features=3, min_samples_leaf=3)
classifier = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=4, min_samples_leaf=7)

att_did = dml_estimators.ATTDID()
att_did.generate_data(n_obs=1000, seed=123)
att_did.setup_model(regressor, CLASSIFIER, n_folds=10)
att_did.fit_model()

print(att_did.model.summary)

late = dml_estimators.LATE()
late.generate_data(n_obs=1000, seed=123)
late.setup_model(regressor, classifier, n_folds=10)
late.fit_model()

print(late.model.summary)
