package org.example

import breeze.linalg.DenseVector
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Success, Try}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


object codeSnippet {

  def main(args: Array[String]) {
    //As the link below states, spark-shell creates by default Spark Context so use that
    //    https://sparkbyexamples.com/spark/sparksession-vs-sparkcontext/?fbclid=IwAR15guKOla8APJa3paaFCNbmfkRRhVp_Il_tOo9F005XpECpj2m1R-uGXkU
    val spark = new SparkContext(new SparkConf().setAppName("codeSnippets").setMaster("local[*]"))

    val baseRdd = spark.textFile("/home/skalogerakis/Projects/CS543/CS543-Assignment2/hw2/dataset.csv")

    println("============1.2===============")
    println("Sanity check count for data points(baseRDD): " + baseRdd.count())
    println("1.2.2 - Top 5 datapoints")
    baseRdd.take(5).foreach(x => println(x))

    println("============1.3===============")
    //Keep everything in cache with cache, as we will use this again and again
//    val parsePointsRdd = baseRdd.foreach(x => inputToLabeled_1_3(x))

    val parsePointsRdd = baseRdd.map(x=>inputToLabeled_1_3(x)).persist()

    println("============1.3.3===============")
    println("Label of the first element: "+parsePointsRdd.first().label)

    println("============1.3.4===============")
    println("Label of the first element: "+parsePointsRdd.first().features)

    println("============1.3.5===============")
    println("Length of the features of the first element: "+parsePointsRdd.first().features.size)

    println("============1.3.6===============")
    println("MAX: "+parsePointsRdd.map(x => x.label).max)
    val minLabel = parsePointsRdd.map(x => x.label).min
    println("MIN: "+minLabel)

    //IMPORTANT: Return again RDD of LabelPoint
    val shiftedPointsRdd = parsePointsRdd.map(x => LabeledPoint(x.label-minLabel,x.features))

    //Debugging
//    shiftedPointsRdd.foreach(println)

    println("============1.4===============")
    println("MAX: "+shiftedPointsRdd.map(x => x.label).max)
    println("MIN: "+shiftedPointsRdd.map(x => x.label).min)

    println("============1.5===============")
    val weights = Array(.8, .1, .1)
    val seed = 42
    val Array(trainData, valData, testData) = shiftedPointsRdd.randomSplit(weights, seed)

    //As 1.5.2 requests, keep all three tables in memory
    trainData.persist()
    valData.persist()
    testData.persist()

    println("============1.5.3===============")
    println("TrainData count: "+trainData.count())
    println("valData count: "+valData.count())
    println("testData count: "+testData.count())
    println("TOTAL count: "+(testData.count()+trainData.count()+valData.count()))

    println("============2.1.1===============")
    val predictionAverage = trainData.map(x => x.label).sum()/trainData.count()
    println("Average Shifted Song year: "+ predictionAverage)

    //2.1.2 -> This function takes any LabelPoint and simply returns the average prediction as computed above.
    //To access predictionAverage variable, define the function in the same object main class
    def baselineModel(lbl: LabeledPoint) : Double ={

       predictionAverage
    }

    //Debugging-> Prints RDD[Double, Double]([prediction,label])
//    trainData.map(x=> (baselineModel(x),x.label)).foreach(println)

    val predsNLabelsTrain: RDD[(Double,Double)] = trainData.map(y => (baselineModel(y),y.label))
    val predsNLabelsVal: RDD[(Double,Double)] = valData.map(y => (baselineModel(y),y.label))
    val predsNLabelsTest: RDD[(Double,Double)] = testData.map(y => (baselineModel(y),y.label))


    println("============2.3.2===============")
    println("RMSE for trainData: "+ calcRmse(predsNLabelsTrain))
    println("RMSE for valData: "+ calcRmse(predsNLabelsVal))
    println("RMSE for testData: "+ calcRmse(predsNLabelsTest))

    println("============3===============")

    //Validated both test cases for 3.1 so gradientSummand works as expected
//    val example_w = DenseVector(1.0, 1.0, 1.0)
//    val example_lp = LabeledPoint(2.0, Vectors.dense(3, 1, 4))
//    println(gradientSummand(example_w, example_lp))

//    val example_w = DenseVector(.24, 1.2, -1.4)
//    val example_lp = LabeledPoint(3.0, Vectors.dense(-1.4, 4.2, 2.1))
//    println(gradientSummand(example_w, example_lp))

    //Validated 3.2 works as expected and returns a tuple of values
//    println(getLabeledPrediction(example_w,example_lp))



    /*********   test lrgd *********/
    val exampleN = 4
    val exampleD = 3
    val exampleData = spark.parallelize(trainData.take(exampleN)).map(lp => LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.slice(0, exampleD))))
    val exampleNumIters = 50
    val (exampleWeights, exampleErrorTrain) = lrgd(exampleData, exampleNumIters)

    println("============3.3.5===============")
    println("Iterations: "+exampleNumIters)
    println("Weights: "+exampleWeights)
    println("Error train: "+exampleErrorTrain)

    println("============3.4===============")
    //TODO

//    val valiData = spark.parallelize(valData.take(exampleN)).map(lp => LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.slice(0, exampleD))))
//    val predsNLabelsVals = valiData.map(x => ((new DenseVector(x.features.toArray) dot exampleWeights), x.label))
//    val calc = calcRmse(predsNLabelsVals)
//    println("RMSE on validation set = "+ calc)

    //todo
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
    val lr=new LinearRegression().setMaxIter(50).setRegParam(0.1).setFitIntercept(true)
    val lrModel = lr.fit(trainDataDF)
    lrModel.evaluate(valDataDF).rootMeanSquaredError
    /***************************************/


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
    val alpha = 0.01 //????????
    val errorTrain = new ListBuffer[Double]
    var weights = new DenseVector(Array.fill[Double](d)(0.0))
    for (i <- 0 until numIter){
      val gradient = trData.map(x => gradientSummand(weights,x)).reduce(_+_) //Compute the gradientSummand and sum all the values together
      val alpha_i = alpha / (n * Math.sqrt(i+1))
      weights -= alpha_i * gradient //Weights change, substract from the previous weight value the gradient
      //update errorTrain
      val predsNLabelsTrain = trData.map(x => getLabeledPrediction(weights,x)) //convert the training set into an RDD of (predictions, labels)
      errorTrain += calcRmse(predsNLabelsTrain)
      println("Iteration RMSE: "+calcRmse(predsNLabelsTrain))
    }
    (weights, errorTrain.toList)
  }

  def getLabeledPrediction(weights: DenseVector[Double], lbl: LabeledPoint): (Double, Double) ={
    val predTmp = weights.dot(DenseVector(lbl.features.toArray))
//    println(predTmp)

    //Return a tuple in scala
    return (predTmp,lbl.label)
  }

  //????????
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
