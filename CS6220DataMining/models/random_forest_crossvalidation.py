from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
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

# Define feature and label columns
initial_feature_columns = [c for c in df.columns if c != 'p_recall']
df = df.withColumnRenamed('p_recall', 'label')

assembler = VectorAssembler(inputCols=initial_feature_columns, outputCol="features")

# Define RandomForest model
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# Create Pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Define a grid of hyperparameters to test:
#   - numTrees: number of trees in the forest.
#   - maxDepth: maximum depth of each tree.
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 50]) \
    .addGrid(rf.maxDepth, [5, 10, 20]) \
    .build()

# Create a cross-validator to perform the hyperparameter tuning
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="label"),
                          numFolds=5)  # Use 5-fold cross-validation

# Train the models and choose the best set of parameters
cvModel = crossval.fit(df)

# Fetch the best model
bestModel = cvModel.bestModel

# Make predictions using the best RandomForest model
predictions = bestModel.transform(df)

# Evaluate the best model using the BinaryClassificationEvaluator
binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = binary_evaluator.evaluate(predictions)

# Evaluate with the MulticlassClassificationEvaluator for other metrics
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
accuracy = multi_evaluator.evaluate(predictions)
precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

# Save results to a text file
results_path = 'optimized_random_forest_results.txt'
with open(results_path, 'w') as file:
    file.write(f"Best Model Parameters:\n")
    file.write(f"Num Trees: {bestModel.stages[-1]._java_obj.getNumTrees()}\n")
    file.write(f"Max Depth: {bestModel.stages[-1]._java_obj.getMaxDepth()}\n")
    file.write(f"Area Under ROC: {auc}\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1 Score: {f1}\n")

print(f"Results saved to {results_path}")

# Stop SparkSession
spark.stop()
