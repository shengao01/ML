import numpy as np
from sklearn.preprocessing import Imputer

data = [[1, 2],
        [np.nan, 3],
        [7, 6]]

imputer = Imputer(missing_values="NaN", strategy="mean", axis=1)
result = imputer.fit_transform(data)
print(result)

