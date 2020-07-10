import matplotlib.pyplot as plt
import pyspark
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pyspark.sql.functions as F
import numpy as np
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, IndexToString, ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from model_performance import *
from processing_data import *

# - PySpark session
spark = pyspark.sql.SparkSession.builder.appName("ZETA Project - DS").getOrCreate()

if __name__ == '__main__':

    # - Original columns are present in 'columns.txt', changes in the column naming made (no whitespaces, '-')
    with open('data/columns.txt') as f_columns:
        columns = [col.strip().replace(' ', '_').replace('-', '_').replace('\'', '') for col in f_columns.readlines()]
    renaming_expression = ['_c{} as {}'.format(i, col) for i, col in enumerate(columns)]

    # - Load data set, rename columns and infer correct schema. Split test/train provided by the author of data
    train_data = (spark.read.option("inferSchema", True)
                  .option("header", False)
                  .csv('data/census-income.data')
                  .selectExpr(renaming_expression))

    test_data = (spark.read.option("inferSchema", True)
                 .option("header", False).csv('data/census-income.test')
                 .selectExpr(renaming_expression))

    # - Show data, printSchema
    train_data.show()
    train_data.printSchema()
    train_data.describe().show()

    # - based on the suggestion, the instance_weight is dropped
    train_data = train_data.drop("instance_weight")
    test_data = test_data.drop("instance_weight")
    columns.remove('instance_weight')

    # - few columns contain "?" - assumed to be missing value
    train_data.select([F.count(F.when(F.isnan(c), c)).alias(c) for c in columns]).show()

    # - print counts of rows and columns
    print("There are {} columns and {} rows in the train set.".format(len(train_data.columns), train_data.count()))

    # - distinct variables for single column...
    train_data.select('migration_code_change_in_msa').distinct().show()

    # - table with all distinct variables in column
    train_data.select(*[F.collect_set(col).alias(col) for col in columns]).show()

    # - EDA
    fig_size = (20, 20)
    train_data.groupBy('class').count().toPandas().plot(kind='bar')

    # - convert for easier visualizations
    train_data_pd = train_data.toPandas()

    # - histograms
    train_data_pd.hist(bins=20, figsize=fig_size)
    train_data_pd.groupby('class').hist(bins=20, figsize=fig_size)

    # - box plots...
    train_data_pd.plot.box(figsize=fig_size)
    train_data_pd.groupby('class').plot.box(figsize=fig_size)

    # - correlation matrix
    plt.figure(figsize=fig_size)
    sns.heatmap(train_data_pd.corr(), annot=True, fmt=".2f", cmap="coolwarm")

    # - distributions for each variables
    cat_bool_col = [col[0] for col in train_data.dtypes if col[1] == 'string' and col[0] != 'class']
    train_set_pd_grouped = dict(list(train_data_pd.groupby(['class'])))
    classes = list(train_set_pd_grouped.keys())
    fig, ax = plt.subplots(len(cat_bool_col), 3, figsize=(20, 150))
    for num_var, var in enumerate(cat_bool_col):
        train_data_pd[var].value_counts().plot(kind="bar", ax=ax[num_var][0], title=var)
    for num_outcome, outcome in enumerate(classes):
        train_set_pd_grouped[outcome][var].value_counts().plot(kind="bar", ax=ax[num_var][1 + num_outcome],
                                                               title='{}: {}'.format(var, outcome))
    plt.show()

    # process data sets
    processed_test_set = process_data_frame(test_data)
    processed_train_set = process_data_frame(train_data)

    # Most of ML algorithms (due to the training process and optimization criterion) have problems with unbalanced data.
    # I would argue that all available algorithms in Spark ML library would suffer in the current scenario
    # (tested just RF) and would learn to predict the majority class (with 95% accuracy in this case) without any data set
    # augmentation.  To complete this task one of the most straightforward data set augmentation technique
    # is used - oversampling of the minority class.

    # compute count of instances per label
    class_instances_counts = processed_train_set.groupBy('has_over_50k').count().collect()

    # add them to dictionary
    class_instances_counts_dict = {row['has_over_50k']: row['count'] for row in class_instances_counts}

    # let's make balanced training set augmented
    augmented_training_set = (processed_train_set
                              .filter(F.col('has_over_50k') == 'Yes')
                              .sample(True,
                                      (class_instances_counts_dict['No'] / class_instances_counts_dict['Yes']) / 2)
                              .union(processed_train_set))

    # Random forests algorithm is used as a classifier. RFs are ensembles of decision trees. Random forests combine
    # many (uncorrelated) decision trees (sensitive to data sample) to reduce the risk of over-fitting.
    # Our data set would not require any further data processing (scaling or normalizing numerical features) except
    # one-hot encoding for categorical features. I went for this one as the cardinality of each remaining categorical
    # feature is low after processing, and the interpretation of something like 'has_high_school_education' seems
    # convenient to me. On top, this algorithm should scale really well with data.

    # list of string columns to be indexes
    string_cols = [col[0] for col in processed_train_set.dtypes if col[1] == 'string' and col[0] != 'has_over_50k']

    # one-hot encoded features
    one_hot_encoded_features = ["{}_encoded".format(col) for col in string_cols]

    # get all features
    num_bool_features = [col[0] for col in processed_train_set.dtypes if
                         col[1] != 'string' and col[0] != 'has_over_50k'] + one_hot_encoded_features

    # create indexer for categorical variables so they can be one-hot encoded
    indexers = [StringIndexer(inputCol=col, outputCol="{}_index".format(col)) for col in string_cols]

    # one-hot encode categorical features
    encoder = OneHotEncoder(inputCols=["{}_index".format(col) for col in string_cols],
                            outputCols=one_hot_encoded_features)

    # assemble all features into feature vector
    features_assembler = VectorAssembler(inputCols=num_bool_features, outputCol="features")

    # Index labels, adding metadata to the label column.
    label_indexer = StringIndexer(inputCol="has_over_50k", outputCol="label").fit(processed_train_set)

    # Convert indexed labels back to original labels.
    label_converter = IndexToString(inputCol="prediction", outputCol="predicted_label", labels=label_indexer.labels)

    # - ChiSQ feature Selection
    selector = ChiSqSelector(numTopFeatures=20, featuresCol="features", outputCol="featuresSel", labelCol="label")

    # - RandomForest model with parameter tuning using cross validation
    rf = RandomForestClassifier(labelCol="label", featuresCol="featuresSel", numTrees=20)

    # - Create ParamGrid for Cross Validation
    rf_param_grid = (ParamGridBuilder()
                     .addGrid(rf.maxDepth, [2, 3, 4, 5, 10, 20])
                     .addGrid(rf.maxBins, [10, 20, 40, 80, 100]).build())

    # - Model Evaluation
    rf_eval = BinaryClassificationEvaluator(labelCol="label")

    # -  Create 5-fold CrossValidator
    cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_param_grid, evaluator=rf_eval, numFolds=5)

    # - chain everything in a Pipeline
    pipeline = Pipeline(stages=indexers + [encoder, features_assembler, label_indexer, selector, cv, label_converter])
    model = pipeline.fit(augmented_training_set)

    # - collect data
    y_score_train, y_pred_train, y_true_train = get_prediction(model.transform(processed_train_set))
    y_score_test, y_pred_test, y_true_test = get_prediction(model.transform(processed_test_set))

    # - compute F1 score, precession and recall
    print("TRAIN")
    print("Precision: {}, recall: {}, F1; {}"
          .format(*precision_recall_fscore_support(y_true_train, y_pred_train, average='binary')))
    print("TEST")
    print("Precision: {}, recall: {}, F1; {}"
          .format(*precision_recall_fscore_support(y_true_test, y_pred_test, average='binary')))

    # - Plot confusion matrix
    plot_confusion_matrix(y_true_train, y_pred_train, normalize=False, classes=[False, True],
                          title="Confusion Matrix for Train Data")
    plot_confusion_matrix(y_true_test, y_pred_test, normalize=False, classes=[False, True],
                          title="Confusion Matrix for Test Data")

    # - Plot ROC curve
    plot_roc_auc(y_true_train, y_score_train, set="Train Sample")
    plot_roc_auc(y_true_test, y_score_test)

    spark.stop()
