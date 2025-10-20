from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

FEATURES = ['Desc length', 'Duration (sec)', 'Post type', 'Day of Week', 'Hour'] #'Permalink'


CAT_FEATURES = ['Post type']
NUM_FEATURES = ['Desc length', 'Duration (sec)', 'Day of Week', 'Hour'] #'Permalink'
TARGETS =  ['Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows']

num_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers = [
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES)
    ],
    remainder="drop"
)