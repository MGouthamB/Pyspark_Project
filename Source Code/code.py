from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, PCA
from pyspark.ml.stat import Correlation
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when
from pyspark.sql.functions import col, count, max as max_
from pyspark.sql.functions import corr, mean, stddev, col
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession

# ----------------------------------------------------------------------------------------------------------------------
# import findspark
# findspark.init()

spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
# ----------------------------------------------------------------------------------------------------------------------
df = spark.read.csv("NF-UQ-NIDS-v2.csv", header=True, inferSchema=True)
# ----------------------------------------------------------------------------------------------------------------------

df.printSchema()


# -----------------------------------------------------------------------------------------------

def null_value_percentage(df):
    total_rows = df.count()
    null_percentage = {}

    for column in df.columns:
        null_count = df.where(col(column).isNull() | isnan(col(column))).count()
        null_percentage[column] = (null_count / total_rows) * 100

    return null_percentage


null_percentage = null_value_percentage(df)
for column, percentage in null_percentage.items():
    print(f"{column}: {percentage}%")


# -----------------------------------------------------------------------------------------------

def calculate_value_percentage(df):
    value_data = []

    for column in df.columns:
        frequency_df = df.groupBy(column).count()
        max_count = frequency_df.agg(max_("count")).collect()[0][0]
        value_values = frequency_df.filter(col("count") == max_count).select(column).collect()
        total_count = df.count()
        value_percentage = (max_count / total_count) * 100
        for row in value_values:
            value_data.append((column, str(row[0]), value_percentage))

    value_df = spark.createDataFrame(value_data, ["Column", "value", "Percentage"])
    return value_df


value_percentage_df = calculate_value_percentage(df)
value_percentage_df.show(value_percentage_df.count(), truncate=False)


# -----------------------------------------------------------------------------------------------

def analyze_value_with_attack(df, value_df):
    analysis_results = []

    for row in value_df.collect():
        column, value_value, _ = row
        filtered_df = df.filter(col(column) == value_value)
        attack_distribution = filtered_df.groupBy('attack').count()
        total_count = filtered_df.count()
        attack_distribution = attack_distribution.withColumn('Percentage', (col('count') / total_count) * 100)
        for attack_row in attack_distribution.collect():
            attack_value, count, percentage = attack_row
            analysis_results.append((column, value_value, attack_value, percentage))

    analysis_df = spark.createDataFrame(analysis_results, ["Column", "value", "Attack", "Percentage"])
    return analysis_df


value_attack_analysis_df = analyze_value_with_attack(df, value_percentage_df)
value_attack_analysis_df.show(value_attack_analysis_df.count(), truncate=False)

# -----------------------------------------------------------------------------------------------

attack_counts = df.groupBy("attack").count()
total_rows = df.count()
attack_percentages = attack_counts.withColumn("Percentage", (col("count") / total_rows) * 100).drop(col("count"))
attack_percentages.sort("Percentage").show(attack_percentages.count())

# ----------------------------------------------------------------------------------------------------------------------

indexer = StringIndexer(inputCol="Attack", outputCol="attackIndex")
df_indexed = indexer.fit(df).transform(df)
indexer = StringIndexer(inputCol="IPV4_SRC_ADDR", outputCol="IPV4_SRC_ADDR_Index")
df_indexed = indexer.fit(df_indexed).transform(df_indexed)
indexer = StringIndexer(inputCol="IPV4_DST_ADDR", outputCol="IPV4_DST_ADDR_Index")
df_indexed = indexer.fit(df_indexed).transform(df_indexed)

df_indexed = df_indexed.drop("IPV4_SRC_ADDR").drop("IPV4_DST_ADDR").drop("Attack")

columns = df_indexed.columns
columns.remove("attackIndex")
columns.remove("Label")

# ----------------------------------------------------------------------------------------------------------------------

stats = []
for column in columns:
    column_stats = df_indexed.select(
        mean(col(column)).alias(column + "_mean"),
        stddev(col(column)).alias(column + "_stddev")
    ).first()

    stats.append((column, column_stats[0], column_stats[1]))

stats_df = spark.createDataFrame(stats, ["Column", "Mean", "StdDev"])

stats_df.show(stats_df.count(), truncate=False)

# -----------------------------------------------------------------------------------------------


Correlations = []
print("Correlations with Label column: ")
for column in columns:
    correlation = df_indexed.select(corr("attackIndex", column)).collect()[0][0]
    # print(f"{column}: {abs(correlation)}")
    if column in ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]: continue
    Correlations.append((column, abs(correlation)))

Correlations.sort(key=lambda x: x[1], reverse=True)
correlation_data = {}
for column, correlation in Correlations:
    correlation_data[column] = correlation
    print(f"{column}: {correlation}")

# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 8))
# plt.barh(list(correlation_data.keys())[1:], list(correlation_data.values())[1:])
# plt.xlabel('Importance Score')
# plt.ylabel('Features')
# plt.title('Feature Importances Pearson Corelation')
# plt.gca().invert_yaxis()
# plt.show()

# -----------------------------------------------------------------------------------------------

columns.remove("SRC_TO_DST_SECOND_BYTES")
df_indexed = df_indexed.drop("SRC_TO_DST_SECOND_BYTES")
columns.remove("DST_TO_SRC_SECOND_BYTES")
df_indexed = df_indexed.drop("DST_TO_SRC_SECOND_BYTES")

df_indexed = df_indexed.dropna()
# df_indexed.describe().show()
columns = df_indexed.columns
columns.remove("attackIndex")
columns.remove("Label")
assembler = VectorAssembler(inputCols=columns, outputCol="features")
df_vector = assembler.transform(df_indexed)

rf = RandomForestClassifier(featuresCol="features", labelCol="Label", maxBins=300000)
model = rf.fit(df_vector)

importances = model.featureImportances
print(importances)

feature_importance_list = [(feature, importance) for feature, importance in zip(columns, importances)]
feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
features, scores = zip(*feature_importance_list)

# -----------------------------------------------------------------------------------------------

important_columns = list(set(list(correlation_data.keys())[1:-30]).union(set(features[:-30])))

print("Top ", len(important_columns), "Columns :")
for i in important_columns:
    print(i)

# -----------------------------------------------------------------------------------------------

filterd_df = df_indexed.select(important_columns)

# ----------------------------------------------------------------------------------------------------------------------
df_pca = filterd_df.drop("Dataset")
df_pca = df_pca.drop("features")
assembler = VectorAssembler(inputCols=important_columns, outputCol="features")
vectorized_data = assembler.transform(df_pca)
pca = PCA(k=len(important_columns), inputCol="features", outputCol="pcaFeatures")
pca_model = pca.fit(vectorized_data)

explained_variance = pca_model.explainedVariance
print(explained_variance)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,6))
# plt.plot(range(1, len(important_columns) + 1), explained_variance,marker='x')
# plt.yscale('log')
# plt.xticks(np.arange(1, len(important_columns) + 1))
# plt.xlabel('Number of Components')
# plt.ylabel('Explained Variance')
# plt.title('Scree Plot')
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------

assembler = VectorAssembler(inputCols=important_columns, outputCol="features")
vectorized_data = assembler.transform(filterd_df)
pca = PCA(k=14, inputCol="features", outputCol="pcaFeatures")
pca_model = pca.fit(vectorized_data)
pca_df = pca_model.transform(vectorized_data)

# ----------------------------------------------------------------------------------------------------------------------

km = KMeans(featuresCol="pcaFeatures").setSeed(1)

evaluator = ClusteringEvaluator(featuresCol='pcaFeatures', metricName='silhouette')

paramGrid = (ParamGridBuilder()
             .addGrid(km.k, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
             .build())

cv = CrossValidator(estimator=km,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=10)

cvModel = cv.fit(pca_df)
print("K means Cross validation Scores:")
for param_map, avg_score in zip(param_maps, avg_scores):
    params = {p.name: v for p, v in param_map.items()}
    print(f"{params} Score: {avg_score}")

# -----------------------------------------------------------------------------------------------


bkm = BisectingKMeans(featuresCol="pcaFeatures").setSeed(1)

evaluator = ClusteringEvaluator(featuresCol='pcaFeatures', metricName='silhouette')

paramGrid = (ParamGridBuilder()
             .addGrid(bkm.k, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
             .build())

cv = CrossValidator(estimator=bkm,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=10)

cvModel = cv.fit(pca_df)
print("Bisecting K means Cross validation Scores:")
for param_map, avg_score in zip(param_maps, avg_scores):
    params = {p.name: v for p, v in param_map.items()}
    print(f"{params} Score: {avg_score}")

# ----------------------------------------------------------------------------------------------------------------------
bkm = BisectingKMeans(featuresCol="pcaFeatures").setK(21).setSeed(1)
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='pcaFeatures', metricName='silhouette')
model = bkm.fit(pca_df)
predictions = model.transform(pca_df)

score = evaluator.evaluate(predictions)
print('Silhouette Score for k = 21 is', score)

count_df = predictions.groupBy('prediction').count()
total_rows = predictions.count()
percentage_df = count_df.withColumn('percentage', (col('count') / total_rows) * 100)
percentage_df.select(["prediction", "percentage"]).sort("percentage", reversed=True).show(100, truncate=False)
count_df = predictions.groupBy('attackIndex').count()
totalrows = predictions.count()
percentage_df = count_df.withColumn('percentage', (col('count') / total_rows) * 100)
percentage_df.select(["attackIndex", "percentage"]).sort("percentage", reversed=True).show(100, truncate=False)
