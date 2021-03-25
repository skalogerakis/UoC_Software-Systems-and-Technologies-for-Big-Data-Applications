package org.example

import breeze.linalg.DenseVector
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Success, Try}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.log4j.{Level, Logger}




object codeSnippet {

  def main(args: Array[String]) {
    //TODO at the end remove that. This removes debugging for better result view
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //As the link below states, spark-shell creates by default Spark Context so use that
    //    https://sparkbyexamples.com/spark/sparksession-vs-sparkcontext/?fbclid=IwAR15guKOla8APJa3paaFCNbmfkRRhVp_Il_tOo9F005XpECpj2m1R-uGXkU
    val spark = new SparkContext(new SparkConf().setAppName("codeSnippets").setMaster("local[*]"))

    /************ Section 1 **********/

    val baseRdd = spark.textFile("/home/skalogerakis/Projects/CS543/CS543-Assignment2/hw2/dataset.csv")

    println("============1.2===============")
    println("Sanity check count for data points(baseRDD): " + baseRdd.count())
    println("\n\n============1.2.2 - Top 5 datapoints============")
    baseRdd.take(5).foreach(x => println(x))

    println("\n\n============1.3===============")


    //Keep everything in cache with persist, as we will use this again and again
    val parsePointsRdd = baseRdd.map(x=>inputToLabeled_1_3(x)).persist()

    println("\n\n============1.3.3===============")
    println("Label of the first element: "+parsePointsRdd.first().label)

    println("\n\n============1.3.4===============")
    println("Features of the first element: "+parsePointsRdd.first().features)

    println("\n\n============1.3.5===============")
    println("Length of the features of the first element: "+parsePointsRdd.first().features.size)

    println("\n\n============1.3.6===============")
    println("MAX: "+parsePointsRdd.map(x => x.label).max)
    val minLabel = parsePointsRdd.map(x => x.label).min
    println("MIN: "+minLabel)

    //IMPORTANT: Return again RDD of LabelPoint
    val shiftedPointsRdd = parsePointsRdd.map(x => LabeledPoint(x.label-minLabel,x.features))

    //Debugging
//    shiftedPointsRdd.foreach(println)

    println("\n\n============1.4===============")
    println("MAX: "+shiftedPointsRdd.map(x => x.label).max)
    println("MIN: "+shiftedPointsRdd.map(x => x.label).min)

    println("\n\n============1.5===============")
    val weights = Array(.8, .1, .1)
    val seed = 42
    //Generate all three datasets in one call, as advised by assignment
    val Array(trainData, valData, testData) = shiftedPointsRdd.randomSplit(weights, seed)

    //As 1.5.2 requests, keep all three tables in memory
    trainData.persist()
    valData.persist()
    testData.persist()

    println("\n\n============1.5.3===============")
    println("TrainData count: "+trainData.count())
    println("valData count: "+valData.count())
    println("testData count: "+testData.count())
    println("TOTAL count: "+(testData.count()+trainData.count()+valData.count()))


    /************ Section 2 **********/

    println("\n\n============2.1.1===============")
    val predictionAverage = trainData.map(x => x.label).sum()/trainData.count()
    println("Average Shifted Song year: "+ predictionAverage)

    //2.1.2 -> This function takes any LabelPoint and simply returns the average prediction as computed above.
    //To access predictionAverage variable, define the function in the same object main class
    def baselineModel(lbl: LabeledPoint) : Double ={

       predictionAverage
    }

    //Debugging-> Prints RDD[Double, Double]([prediction,label])
//    trainData.map(x=> (baselineModel(x),x.label)).foreach(println)

    //Transform everything in the desired format of RMSE function. Define type just to make sure it works as expected
    val predsNLabelsTrain: RDD[(Double,Double)] = trainData.map(y => (baselineModel(y),y.label))
    val predsNLabelsVal: RDD[(Double,Double)] = valData.map(y => (baselineModel(y),y.label))
    val predsNLabelsTest: RDD[(Double,Double)] = testData.map(y => (baselineModel(y),y.label))


    println("\n\n============2.3.2===============")
    println("RMSE for trainData: "+ calcRmse(predsNLabelsTrain))
    println("RMSE for valData: "+ calcRmse(predsNLabelsVal))
    println("RMSE for testData: "+ calcRmse(predsNLabelsTest))

    /************ Section 3 **********/

    println("\n\n============3===============")

    //Validated both test cases for 3.1 so gradientSummand works as expected
    println("\n\n============3.1===============")
    val example_w = DenseVector(1.0, 1.0, 1.0)
    val example_lp = LabeledPoint(2.0, Vectors.dense(3, 1, 4))
    println("Expected output for gradient summand function [18.0, 6.0, 24.0]. Function output: "+gradientSummand(example_w, example_lp))

//    val example_w = DenseVector(.24, 1.2, -1.4)
//    val example_lp = LabeledPoint(3.0, Vectors.dense(-1.4, 4.2, 2.1))
//    println(gradientSummand(example_w, example_lp))

    //Validated 3.2 works as expected and returns a tuple of values
//    println(getLabeledPrediction(example_w,example_lp))

    // We change the code for 3.3.2. RMSE goes to infinity. Values for 3.3.2 can be found in the report. lrgd function changed for the purposes of 3.3.5
    //Example provided by assignment
//    val exampleN = 4
//    val exampleD = 3
//    val exampleData = spark.parallelize(trainData.take(exampleN)).map(lp => LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.slice(0, exampleD))))
//    val exampleNumIters = 50
//    val (exampleWeights, exampleErrorTrain) = lrgd(exampleData, exampleNumIters)

    val NumIters = 50
    val (fWeights, exampleErrorTrain) = lrgd(trainData, NumIters)

    println("\n\n============3.3.5===============")
    println("Iterations: "+NumIters)
    println("Weights: "+fWeights)
    println("Error train: "+exampleErrorTrain.take(50))


    println("\n\n============3.4===============")

    val predsNLabelsVals = valData.map(x => getLabeledPrediction(fWeights,x))

    val calc = calcRmse(predsNLabelsVals)
    println("RMSE on validation set = "+ calc)


    /************ Section 4 **********/
    /********** MLLib + grid search*****************************/
    import org.apache.spark.ml.regression.LinearRegression
    import org.apache.spark.ml.linalg.{Vectors => MLVectors}
    import org.apache.spark.ml.feature.{LabeledPoint => MLabeledPoint}
    val sqlContext = new org.apache.spark.sql.SQLContext(spark)
    import sqlContext.implicits._


    /*********************RDD conversion to Dataframe*****************/
    val trainDataDF = trainData.map(lp => MLabeledPoint(lp.label, MLVectors.dense(lp.features.toArray))).toDF
    val valDataDF = valData.map(lp => MLabeledPoint(lp.label, MLVectors.dense(lp.features.toArray))).toDF
    val testDataDF = testData.map(lp => MLabeledPoint(lp.label, MLVectors.dense(lp.features.toArray))).toDF
    /*******************************************************************/


    /******Linear Regression Demo*********/
    val lr=new LinearRegression().setMaxIter(50).setRegParam(0.1).setFitIntercept(true) //Initiates Linear Regression and fits model using 50 iter
    val lrModel = lr.fit(trainDataDF)
//    lrModel.evaluate(valDataDF).rootMeanSquaredError

    println("\n\n============4===============")

    println("\n\n============4.1.1===============")
    println("Coefficients: " +lrModel.coefficients )
    println("Intercept: "+lrModel.intercept)

    println("\n\n============4.1.2===============")
    //TODO ASK IF THIS IS CORRECT????
    val rmseVal1 = lrModel.evaluate(valDataDF).rootMeanSquaredError
    println("RMSE on validation set: "+rmseVal1)

    println("\n\n============4.1.3===============")
    println("Transformed validation set - First 10 predictions")
    lrModel.transform(valDataDF).show(10,false) //Transform validation set on existing model

    println("\n\n============4.2===============")

    println("============4.2.1===============")

    //Simply change setRegParam property with the correspond values and compare the results
    val lr_2 =new LinearRegression().setMaxIter(50).setRegParam(1e-10).setFitIntercept(true) //Initiates Linear Regression and fits model using 50 iter
    val lrModel_2 = lr_2.fit(trainDataDF)
    val rmseVal2 = lrModel_2.evaluate(valDataDF).rootMeanSquaredError

    val lr_3 =new LinearRegression().setMaxIter(50).setRegParam(1e-5).setFitIntercept(true) //Initiates Linear Regression and fits model using 50 iter
    val lrModel_3 = lr_3.fit(trainDataDF)
    val rmseVal3 = lrModel_3.evaluate(valDataDF).rootMeanSquaredError

    val lr_4 =new LinearRegression().setMaxIter(50).setRegParam(1).setFitIntercept(true) //Initiates Linear Regression and fits model using 50 iter
    val lrModel_4 = lr_4.fit(trainDataDF)
    val rmseVal4 = lrModel_4.evaluate(valDataDF).rootMeanSquaredError

    println("RMSE comparison per regularization parameter")
    println("For regParam 0,1: "+rmseVal1)
    println("For regParam 1e-10: "+rmseVal2)
    println("For regParam 1e-5: "+rmseVal3)
    println("For regParam 1: "+rmseVal4)


    /************ Section 5 **********/

    println("\n\n============5===============")

    //5.1.1 transform crossTrainDataRDD to DF
    val crossTrainDataRDD = trainData.map(lp => quadFeatures(lp))
    val crossTrainDataDF = crossTrainDataRDD.map(lp => MLabeledPoint(lp.label, MLVectors.dense(lp.features.toArray))).toDF  //As provided by assignment and used in section 4
//    println("Sanity check for crossTrainDataDF: "+crossTrainDataDF.show(10))

    //5.1.2 Now transform validation and tests in similar fashion
    val crossValDataRDD = valData.map(lp => quadFeatures(lp))
    val crossValDataDF = crossValDataRDD.map(lp => MLabeledPoint(lp.label, MLVectors.dense(lp.features.toArray))).toDF

    val crossTestDataRDD = testData.map(lp => quadFeatures(lp))
    val crossTestDataDF = crossTestDataRDD.map(lp => MLabeledPoint(lp.label, MLVectors.dense(lp.features.toArray))).toDF

    //5.2 Build new model with the specs provided by assignment
    val lr_final=new LinearRegression().setMaxIter(500).setRegParam(1e-10).setFitIntercept(true) //Initiates Linear Regression and fits model using 50 iter
    val lrModel_final = lr_final.fit(crossTrainDataDF)


    println("============5.4.1===============")
    //5.3 Find the RMSE of the new model
    println("New model RMSE: "+ lrModel_final.evaluate(crossValDataDF).rootMeanSquaredError)

    //5.4
    println("Baseline model RMSE: "+ calcRmse(predsNLabelsVal)) //val predsNLabelsVal defined at the start and used baseline model

    println("============5.4.2===============")
    lrModel_final.transform(crossTestDataDF).select("prediction").show(50)


    /**** Use of pipelines ******************************/
    //Following the pipeline example https://spark.apache.org/docs/latest/ml-pipeline.html
    import org.apache.spark.ml.feature.PolynomialExpansion
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.evaluation.RegressionEvaluator

    val numIters = 500
    val reg = 1e-10
    val alpha = .2
    val useIntercept = true
    val polynomial_expansion = (new PolynomialExpansion).setInputCol("features").setOutputCol("polyFeatures").setDegree(2)
    val lr3 = new LinearRegression()
    lr3.setMaxIter(numIters).setRegParam(reg).setElasticNetParam(alpha).setFitIntercept(useIntercept).setFeaturesCol("polyFeatures")

    val pipeline = new Pipeline()
    pipeline.setStages(Array(polynomial_expansion,lr3)) //there are two stages here that you have to set.

    //TODO ask about that as well
    val model= pipeline.fit(trainDataDF) //need to fit. Use the train Dataframe defined at the beginning of section 4
    val predictionsDF=model.transform(testDataDF) //Produce predictions on the test set. Use method transform.
    val evaluator = new RegressionEvaluator()
    evaluator.setMetricName("rmse")
    val rmseTestPipeline = evaluator.evaluate(predictionsDF)
    println("RMSE final model test "+rmseTestPipeline)

  }

  /***** Quadratic Feature extraction for 5.1 ********/
  implicit class Crossable[X](xs: Traversable[X]) {
    def cross[Y](ys: Traversable[Y]) = for { x <- xs; y <- ys } yield (x, y)
  }

  def quadFeatures(lp: LabeledPoint) = {
    val crossFeatures = lp.features.toArray.toList cross lp.features.toArray.toList
    val sqFeatures = crossFeatures.map(x => x._1 * x._2).toArray
    LabeledPoint(lp.label, Vectors.dense(sqFeatures))
  }


  /*MODIFIED lrgd. The function takes as input an
    RDD of LabeledPoints and the number of Iterations and returns the model
    parameters, and the list of rmse for each iteration. Fill with code the
    question marks (?) and debug.
    */
  import scala.collection.mutable.ListBuffer
  def lrgd(trData: RDD[LabeledPoint], numIter: Int): (DenseVector[Double], List[Double]) = {
    val n = trData.count
    val d = trData.first.features.size
    val alpha = 0.01
    println("Alpha Value: "+alpha)
    val errorTrain = new ListBuffer[Double]
    var weights = new DenseVector(Array.fill[Double](d)(0.0))

    for (i <- 0 until numIter){
      val gradient = trData.map(x => gradientSummand(weights,x)).reduce(_+_) //Compute the gradientSummand and sum all the values together
      val alpha_i = alpha / (n * Math.sqrt(i+1))
      weights -= alpha_i * gradient //Weights change, following the equation. To compute w(i+1) we must use w(i)
      //update errorTrain
      val predsNLabelsTrain = trData.map(x => getLabeledPrediction(weights,x)) //convert the training set into an RDD of (predictions, labels). Use the function created previously
      errorTrain += calcRmse(predsNLabelsTrain)
      println("Iteration RMSE: "+calcRmse(predsNLabelsTrain))
    }
    (weights, errorTrain.toList)
  }

  //Takes a dense vector of weights and a labeledPoint and returns (prediction,label)
  def getLabeledPrediction(weights: DenseVector[Double], lbl: LabeledPoint): (Double, Double) ={
    //Make prediction by computing the dot product between weights and the features of a labeled point
    val predTmp = weights.dot(DenseVector(lbl.features.toArray))
//    println(predTmp)

    //Return a tuple in scala
    return (predTmp,lbl.label)
  }

  def gradientSummand(weights: DenseVector[Double], lp: LabeledPoint): DenseVector[Double] ={
    //Basically follow the example given to reproduce the correct result
    //gradient_summand = (dot([1 1 1], [3 1 4]) - 2) * [3 1 4] = (8 - 2) * [3 1 4] = [18 6 24]
    //However, in order to use dot function without problem, LabelPoint features need casting in DenseVector[Double] otherwise the compiler complains
    return (weights.dot(DenseVector(lp.features.toArray)) - lp.label) * DenseVector(lp.features.toArray)

  }


  //2.2 Takes input an RDD[Double, Double] and returns Double which corresponds to rootMeanSquaredError
  def calcRmse(prd: RDD[(Double, Double)]): Double ={

    val regressionMetrics = new RegressionMetrics(prd)
//    println(s"RMSE = ${regressionMetrics.rootMeanSquaredError}")
    regressionMetrics.rootMeanSquaredError

  }

  def inputToLabeled_1_3(str: String): LabeledPoint ={

    //Could simply map(s=>s.toDouble) as we know that our dataset contains valid double values. To make everything bulletproof apply the idea in the link below.
    // Check if our value can become double and only then convert it to double. Avoid exceptions
    //    https://stackoverflow.com/questions/33848755/convert-a-scala-list-of-strings-into-a-list-of-doubles-while-discarding-unconver
    val attrList = str.split(",").map( s => Try(s.toDouble) ).collect { case Success(x) => x }.toList
//    val attrList = str.split(",").map( s => s.toDouble).toList

//    println(attrList)

    //Label is the first value(using head list function) and features are the rest(tail function takes all but first element)
    LabeledPoint(attrList.head, Vectors.dense(attrList.tail.toArray))

  }




}
