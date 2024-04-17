from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("OptimizedRandomForest") \
    .config("spark.executor.memory", "6g") \
    .config("spark.driver.memory", "6g") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

# Load data
data_path = 'dataset_final_downsample_with_session_correct.csv'
df = spark.read.csv(data_path, header=True, inferSchema=True).repartition(200)


# Define feature and label columns initially
initial_feature_columns = [c for c in df.columns if c != 'p_recall']
assembler = VectorAssembler(inputCols=initial_feature_columns, outputCol="features")

# Define RandomForest model
rf = RandomForestClassifier(featuresCol="features", labelCol="p_recall")

# Create Pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Train model
model = pipeline.fit(df)

# Evaluate model
binary_evaluator = BinaryClassificationEvaluator(labelCol="p_recall")
auc = binary_evaluator.evaluate(model.transform(df), {binary_evaluator.metricName: "areaUnderROC"})

# Extract feature importances and filter features
feature_importances = model.stages[-1].featureImportances.toArray()
important_features = [feature for feature, importance in zip(initial_feature_columns, feature_importances) if importance > 0]

# Redefine assembler with important features only
assembler = VectorAssembler(inputCols=important_features, outputCol="features")

# Redefine pipeline with new assembler
pipeline = Pipeline(stages=[assembler, rf])

# Retrain model with filtered features
model = pipeline.fit(df)

# Make new predictions
predictions = model.transform(df)

# Re-evaluate model
accuracy = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
multi_evaluator = MulticlassClassificationEvaluator(labelCol="p_recall", metricName="accuracy")
precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

# Save results to a text file
results_path = 'optimized_random_forest_results.txt'
with open(results_path, 'w') as file:
    file.write(f"Area Under ROC: {auc}\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1 Score: {f1}\n")

print(f"Results saved to {results_path}")

# Stop SparkSession
spark.stop()
