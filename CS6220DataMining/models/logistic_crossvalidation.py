from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col
import matplotlib.pyplot as plt

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("OptimizedLogisticRegression") \
    .config("spark.executor.memory", "6g") \
    .config("spark.driver.memory", "6g") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

# Load data
data_path = '../dataset_final.csv'
df = spark.read.csv(data_path, header=True, inferSchema=True)
df = df.withColumn('label', (col('p_recall') > 0.5).cast('int'))

# Define feature columns and assembler
feature_columns = [c for c in df.columns if c not in ['p_recall', 'label']]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Feature scaling
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Logistic Regression model
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label")

# Create a Pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Set up the parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 0.8])  # Test L2, Elastic Net, and L1 penalties
    .addGrid(lr.regParam, [0.01, 0.05, 0.1])  # Test different strengths of regularization
    .build()

# Set up the CrossValidator
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=BinaryClassificationEvaluator(labelCol="label"),
                    numFolds=5)  # Use 5-fold cross-validation

# Fit the model using CrossValidator
cvModel = cv.fit(df)

# Make predictions on the entire dataset using the best model found
predictions = cvModel.bestModel.transform(df)

# Evaluate the best model
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
accuracy = multi_evaluator.evaluate(predictions)
precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

# Save the results to a text file
results_path = '../optimized_logistic_regression_results.txt'
with open(results_path, 'w') as file:
    file.write(f"Area Under ROC: {roc_auc}\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1 Score: {f1}\n")

# Plot and save the objective history
trainingSummary = cvModel.bestModel.stages[-1].summary
objectiveHistory = trainingSummary.objectiveHistory
plt.figure(figsize=(8, 5))
plt.plot(objectiveHistory)
plt.xlabel('Iteration')
plt.ylabel('Log-Loss')
plt.title('Logistic Regression Log-Loss Curve')
plt.grid(True)
plt.savefig('log_likelihood_curve.png')
plt.show()

# Stop the Spark session
spark.stop()
